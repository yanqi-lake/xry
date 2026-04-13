#!/usr/bin/env python3
"""
Odorant Mixture Perceptual Similarity Prediction
============================================
Based on: Snitz et al. (2013) PLoS Comput Biol 9(9): e1003184
Predicting Odor Perceptual Similarity from Odor Structure

METHODOLOGY (from paper):
1. Normalize each Dragon descriptor to [0,1] range:
   vn = (v - min(ld)) / (max(ld) - min(ld))

2. Create mixture total vector by summing component descriptors:
   MixVector = Σ(component_i descriptors)
   Then normalize by dividing by its norm to eliminate effect of number of components

3. Calculate angle between mixture vectors:
   angle = arccos((U·V) / (|U||V|)) in degrees
   
4. Information entropy of mixture:
   H = -Σ(p * log2(p)) where p is normalized descriptor value

5. Relative perceptual complexity:
   RPC = entropy / log2(n_descriptors)

REQUIREMENTS:
- pandas
- numpy
- openpyxl (for reading Excel files)

Install: pip install pandas numpy openpyxl
"""

import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "openpyxl", "-q"])
    print("Installation complete.\n")

def main():
    # Try to import, install if missing
    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        install_requirements()
        import pandas as pd
        import numpy as np
    
    print("=" * 70)
    print("Odorant Mixture Perceptual Similarity Prediction")
    print("Based on Snitz et al. (2013) PLoS Comput Biology")
    print("=" * 70)
    
    # File paths (assuming files are in same directory as script)
    descriptor_file = '13-descriptors-dragon.xlsx'
    mixture_file = 'mixture-components.xlsx'
    
    # Check files exist
    if not os.path.exists(descriptor_file):
        print(f"Error: {descriptor_file} not found!")
        print("Please ensure Excel files are in the same directory as this script.")
        return
    
    # =========================================================================
    # STEP 1: Load molecular descriptor data
    # =========================================================================
    print("\n[1] Loading molecular descriptors...")
    desc_df = pd.read_excel(descriptor_file)
    print(f"    Loaded {desc_df.shape[0]} molecules, {desc_df.shape[1]} descriptors")
    
    # Get molecule IDs (first column should be MOL_ID)
    mol_col = desc_df.columns[0]
    molecule_ids = desc_df[mol_col].values
    descriptor_names = desc_df.columns[1:].tolist()
    
    # Get descriptor matrix (skip molecule ID column)
    descriptor_matrix = desc_df.iloc[:, 1:].values.astype(float)
    
    # Replace -999 (missing value marker) with NaN
    descriptor_matrix[descriptor_matrix == -999] = np.nan
    
    print(f"    Descriptor names: {descriptor_names[:10]}...")
    print(f"    First molecule: {molecule_ids[0]}")
    
    # =========================================================================
    # STEP 2: Load mixture composition data
    # =========================================================================
    print("\n[2] Loading mixture composition data...")
    
    if os.path.exists(mixture_file):
        mix_df = pd.read_excel(mixture_file)
        print(f"    Mixture data shape: {mix_df.shape}")
        print(f"    Columns: {mix_df.columns.tolist()}")
        print(mix_df)
    else:
        print(f"    Warning: {mixture_file} not found.")
        print("    Using manual component specification...")
        mix_df = None
    
    # =========================================================================
    # STEP 3: Normalize descriptors to [0,1] range
    # =========================================================================
    print("\n[3] Normalizing descriptors to [0,1] range...")
    
    # For each descriptor (column), find min and max
    col_min = np.nanmin(descriptor_matrix, axis=0)
    col_max = np.nanmax(descriptor_matrix, axis=0)
    col_range = col_max - col_min
    
    # Avoid division by zero ( descriptors with constant values)
    col_range[col_range == 0] = 1
    
    # Normalize: vn = (v - min) / (max - min)
    normalized_matrix = (descriptor_matrix - col_min) / col_range
    
    # Replace NaN with 0
    normalized_matrix = np.nan_to_num(normalized_matrix, nan=0.0)
    
    print(f"    Normalized matrix shape: {normalized_matrix.shape}")
    print(f"    Value range: [{normalized_matrix.min():.3f}, {normalized_matrix.max():.3f}]")
    
    # =========================================================================
    # STEP 4: Create mixture vectors and calculate similarity
    # =========================================================================
    print("\n[4] Creating mixture vectors and calculating similarity...")
    
    # Function to create mixture vector from component indices
    def create_mixture_vector(component_indices, desc_matrix):
        """
        Create mixture vector by summing component descriptors,
        then normalizing by dividing by its norm.
        """
        component_vectors = desc_matrix[component_indices]
        mixture_vector = np.sum(component_vectors, axis=0)
        
        # Normalize by dividing by its norm
        norm = np.linalg.norm(mixture_vector)
        if norm > 0:
            mixture_vector = mixture_vector / norm
            
        return mixture_vector
    
    # Function to calculate angle between two vectors
    def vector_angle(vec1, vec2):
        """Calculate angle between two vectors in degrees."""
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        cos_angle = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        return np.degrees(angle_rad)
    
    # Function to calculate information entropy
    def information_entropy(vector):
        """Calculate Shannon information entropy."""
        abs_vec = np.abs(vector)
        total = np.sum(abs_vec)
        
        if total == 0:
            return 0.0
            
        probability = abs_vec / total
        entropy = 0.0
        
        for p in probability:
            if p > 0:
                entropy -= p * np.log2(p)
                
        return entropy
    
    # Function to calculate relative perceptual complexity
    def relative_perceptual_complexity(entropy, n_descriptors):
        """Calculate relative perceptual complexity."""
        max_entropy = np.log2(n_descriptors)
        return entropy / max_entropy if max_entropy > 0 else 0
    
    # =========================================================================
    # EXAMPLE CALCULATIONS
    # =========================================================================
    print("\n[5] Example calculations...")
    
    # Create example mixtures
    # Mix 1: first 3 molecules (indices 0,1,2)
    # Mix 2: molecules 0,2,4 (some overlap)
    mix1_indices = [0, 1, 2]
    mix2_indices = [0, 2, 4] if normalized_matrix.shape[0] > 4 else [0, 1, 2]
    
    mix1_vector = create_mixture_vector(mix1_indices, normalized_matrix)
    mix2_vector = create_mixture_vector(mix2_indices, normalized_matrix)
    
    # Calculate angle
    angle = vector_angle(mix1_vector, mix2_vector)
    
    # Convert angle to similarity (0-100 scale)
    similarity = ((180 - angle) / 180) * 100
    
    print(f"\n    Mixture 1 components: {mix1_indices}")
    print(f"    Mixture 2 components: {mix2_indices}")
    print(f"    Vector angle: {angle:.2f}°")
    print(f"    Perceptual similarity: {similarity:.2f}/100")
    
    # Calculate entropies
    entropy1 = information_entropy(mix1_vector)
    entropy2 = information_entropy(mix2_vector)
    
    n_desc = normalized_matrix.shape[1]
    complexity1 = relative_perceptual_complexity(entropy1, n_desc)
    complexity2 = relative_perceptual_complexity(entropy2, n_desc)
    
    print(f"\n    Mixture 1 entropy: {entropy1:.4f}")
    print(f"    Mixture 2 entropy: {entropy2:.4f}")
    print(f"    Mixture 1 complexity: {complexity1:.4f}")
    print(f"    Mixture 2 complexity: {complexity2:.4f}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("INTERPRETATION GUIDE:")
    print("=" * 70)
    print("""
KEY METRICS:
1. Vector Angle (0° - 180°)
   - Smaller angle = more similar odor perception
   - 0° = identical, 180° = completely different
   
2. Perceptual Similarity Score (0 - 100)
   - Higher score = more similar perception
   - Based on angle: similarity = (180 - angle) / 180 * 100

3. Information Entropy (bits)
   - Higher entropy = more complex/informative mixture
   - Range: 0 to log2(n_descriptors)

4. Relative Perceptual Complexity (0 - 1)
   - ratio of actual entropy to maximum possible entropy
   - Higher = more complex odor profile

TO USE WITH YOUR DATA:
1. Edit the mixture component indices (e.g., mix1_indices = [0,1,5,8])
2. Re-run the script
3. Compare different mixtures

EXAMPLE COMPARISON:
- Mix1 (EGM): Acetophenone, Eugenol, Guaiacol, Methyl anthranilate, Cinnamaldehyde
- Mix2 (PCE): Pentyl valerate, 2-Nonanol, 2-Octanone

Simply replace the component indices in the code!
""")

if __name__ == "__main__":
    main()