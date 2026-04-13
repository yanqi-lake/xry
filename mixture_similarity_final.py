#!/usr/bin/env python3
"""
Odorant Mixture Perceptual Similarity Prediction
================================================
Based on: Snitz et al. (2013) PLoS Comput Biol 9(9): e1003184
Predicting Odor Perceptual Similarity from Odor Structure

Paper URL: https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003184

============================================
INSTALLATION (run this in your terminal):
============================================
pip install pandas numpy openpyxl scipy

============================================
USAGE:
============================================
python mixture_similarity.py

Then edit the component indices in the code to match your mixture data.

============================================
METHODOLOGY:
============================================
1. NORMALIZE DESCRIPTORS to [0,1]:
   vn = (v - min(ld)) / (max(ld) - min(ld))

2. CREATE MIXTURE VECTOR:
   MixVector = sum(component descriptors)
   Then normalize by dividing by its norm:
   MixVector_normalized = MixVector / ||MixVector||

3. CALCULATE ANGLE (perceptual similarity metric):
   angle = arccos((U·V) / (||U|| × ||V||)) × (180/π)  [in degrees]
   
4. INFORMATION ENTROPY:
   H = -Σ(p × log2(p))
   
5. RELATIVE PERCEPTUAL COMPLEXITY:
   RPC = entropy / log2(n_descriptors)

============================================
"""

import pandas as pd
import numpy as np
import os

def main():
    print("=" * 70)
    print("Odorant Mixture Perceptual Similarity Prediction")
    print("Based on Snitz et al. (2013) PLoS Comput Biology")
    print("=" * 70)
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    DESCRIPTOR_FILE = '13-descriptors-dragon.xlsx'
    MIXTURE_FILE = 'mixture-components.xlsx'
    AUTO_ANALYZE = True  # Set to True to analyze all mixtures automatically
    
    # Manual mixture components (used if AUTO_ANALYZE = False)
    MIX1_COMPONENTS = [0, 1, 2, 3, 4]      # Example: first 5 molecules
    MIX2_COMPONENTS = [5, 6, 7, 8, 9]      # Example: next 5 molecules
    
    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    print("\n[Step 1] Loading data...")
    
    if not os.path.exists(DESCRIPTOR_FILE):
        print(f"ERROR: {DESCRIPTOR_FILE} not found!")
        return
    
    # Load descriptors
    df_desc = pd.read_excel(DESCRIPTOR_FILE, header=None, skiprows=3)
    df_desc.columns = ['No', 'MOL_ID'] + [f'D{i}' for i in range(df_desc.shape[1]-2)]
    df_desc = df_desc.drop(columns=['No'])
    
    n_molecules, n_descriptors = df_desc.shape[0], df_desc.shape[1] - 1
    
    print(f"  Loaded: {n_molecules} molecules, {n_descriptors} descriptors")
    print(f"  Molecule names: {list(df_desc['MOL_ID'].values)}")
    
    # Load mixture data
    if os.path.exists(MIXTURE_FILE):
        df_mix = pd.read_excel(MIXTURE_FILE, header=0)
        df_mix.columns = ['Similarity Level', 'Mixture', 'Component 1', 'Component 2', 'Component 3', 'Component 4']
    else:
        print(f"  Warning: {MIXTURE_FILE} not found")
        df_mix = None
    
    # =========================================================================
    # STEP 2: Normalize descriptors
    # =========================================================================
    print("\n[Step 2] Normalizing descriptors to [0,1]...")
    
    desc_cols = df_desc.columns[1:]
    desc_matrix = df_desc[desc_cols].values.astype(float)
    desc_matrix[desc_matrix == -999] = np.nan
    
    col_min = np.nanmin(desc_matrix, axis=0)
    col_max = np.nanmax(desc_matrix, axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    
    norm_matrix = (desc_matrix - col_min) / col_range
    norm_matrix = np.nan_to_num(norm_matrix, nan=0.0)
    
    print(f"  Normalized shape: {norm_matrix.shape}")
    
    # =========================================================================
    # Helper functions
    # =========================================================================
    def create_mixture_vector(indices, desc_matrix):
        if max(indices) >= desc_matrix.shape[0]:
            indices = [i for i in indices if i < desc_matrix.shape[0]]
        mix_vec = np.sum(desc_matrix[indices], axis=0)
        norm = np.linalg.norm(mix_vec)
        if norm > 0:
            mix_vec = mix_vec / norm
        return mix_vec
    
    def vector_angle(vec1, vec2):
        dot = np.dot(vec1, vec2)
        n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if n1 == 0 or n2 == 0:
            return 0.0
        cos_angle = np.clip(dot / (n1 * n2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    
    def calc_entropy(vector):
        p = np.abs(vector)
        total = np.sum(p)
        if total == 0:
            return 0.0
        p = p / total
        return -np.sum(p[p > 0] * np.log2(p[p > 0]))
    
    # =========================================================================
    # ANALYSIS
    # =========================================================================
    if AUTO_ANALYZE and df_mix is not None:
        # Auto analyze all mixtures
        print("\n[Step 3] Analyzing all mixtures...")
        
        mol_names = df_desc['MOL_ID'].values
        mol_to_idx = {name: idx for idx, name in enumerate(mol_names)}
        
        # Parse mixtures
        mixtures = {}
        component_cols = ['Component 1', 'Component 2', 'Component 3', 'Component 4']
        
        for idx, row in df_mix.iterrows():
            mix_name = row['Mixture']
            components = []
            for col in component_cols:
                if col in df_mix.columns and pd.notna(row[col]):
                    mol_name = str(row[col]).strip()
                    if mol_name in mol_to_idx:
                        components.append(mol_to_idx[mol_name])
                    else:
                        print(f"  Warning: '{mol_name}' not found")
            
            if components:
                mixtures[mix_name] = components
        
        print(f"  Parsed {len(mixtures)} mixtures:")
        for name, comps in mixtures.items():
            mol_list = [mol_names[i] for i in comps]
            print(f"    {name}: {mol_list}")
        
        # Create vectors and compute similarity matrix
        print("\n" + "=" * 70)
        print("PAIRWISE PERCEPTUAL SIMILARITY MATRIX")
        print("=" * 70)
        
        mix_vectors = {}
        for mix_name, comp_indices in mixtures.items():
            mix_vec = create_mixture_vector(comp_indices, norm_matrix)
            mix_vectors[mix_name] = mix_vec
        
        mix_names_list = list(mix_vectors.keys())
        n = len(mix_names_list)
        
        # Print matrix
        print(f"\n{'Mixture':<12}", end="")
        for name in mix_names_list:
            print(f"{name:<12}", end="")
        print()
        print("-" * (12 * (n + 1)))
        
        pair_results = []
        for i, name1 in enumerate(mix_names_list):
            print(f"{name1:<12}", end="")
            for j, name2 in enumerate(mix_names_list):
                if i == j:
                    sim = 100.0
                else:
                    angle = vector_angle(mix_vectors[name1], mix_vectors[name2])
                    sim = ((180 - angle) / 180) * 100
                pair_results.append((name1, name2, sim))
                print(f"{sim:>10.2f}  ", end="")
            print()
        
        # Save results to Excel
        print("\n[Step 4] Saving results...")
        results_df = pd.DataFrame(pair_results, columns=['Mixture 1', 'Mixture 2', 'Similarity'])
        results_df.to_excel('similarity_results.xlsx', index=False)
        print(f"  Results saved to: similarity_results.xlsx")
        
    else:
        # Manual analysis (two mixtures)
        print("\n[Step 3] Creating mixture vectors...")
        
        mix1_vec = create_mixture_vector(MIX1_COMPONENTS, norm_matrix)
        mix2_vec = create_mixture_vector(MIX2_COMPONENTS, norm_matrix)
        
        print(f"  Mixture 1: components {MIX1_COMPONENTS}")
        print(f"  Mixture 2: components {MIX2_COMPONENTS}")
        
        print("\n[Step 4] Calculating perceptual similarity...")
        
        angle = vector_angle(mix1_vec, mix2_vec)
        similarity = ((180 - angle) / 180) * 100
        
        print(f"  Vector angle: {angle:.2f}°")
        print(f"  Perceptual similarity: {similarity:.2f}/100")
        
        print("\n[Step 5] Calculating entropy and complexity...")
        
        entropy1 = calc_entropy(mix1_vec)
        entropy2 = calc_entropy(mix2_vec)
        
        max_entropy = np.log2(n_descriptors)
        complexity1 = entropy1 / max_entropy
        complexity2 = entropy2 / max_entropy
        
        print(f"  Mixture 1 entropy: {entropy1:.4f}, complexity: {complexity1:.4f}")
        print(f"  Mixture 2 entropy: {entropy2:.4f}, complexity: {complexity2:.4f}")
        
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"  Mixture 1: {MIX1_COMPONENTS}")
        print(f"  Mixture 2: {MIX2_COMPONENTS}")
        print(f"  Similarity: {similarity:.2f}/100")


if __name__ == "__main__":
    main()