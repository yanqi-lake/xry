#!/usr/bin/env python3
"""
Odorant Mixture Perceptual Similarity Prediction
Based on: Snitz et al. (2013) PLoS Comput Biol
 Predicting Odor Perceptual Similarity from Odor Structure

This program:
1. Loads Dragon molecular descriptors
2. Loads mixture composition data
3. Normalizes descriptors to [0,1] range
4. Creates mixture total vectors (sum of component vectors, normalized)
5. Computes angle between two mixture vectors (perceptual similarity metric)
6. Computes information entropy of mixture descriptors
7. Computes relative perceptual complexity
"""

import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Try to load pandas, otherwise use openpyxl directly
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    import zipfile
    import xml.etree.ElementTree as ET

def excel_to_dataframe_nopandas(filepath):
    """Parse Excel file without pandas using zipfile and xml"""
    data = {}
    
    with zipfile.ZipFile(filepath, 'r') as z:
        # Get shared strings
        try:
            with z.open('xl/sharedStrings.xml') as f:
                content = f.read().decode('utf-8')
            ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
            root = ET.fromstring(content)
            strings = []
            for si in root.findall('.//s:si', ns):
                t = si.find('s:t', ns)
                if t is not None:
                    strings.append(t.text if t.text else '')
                else:
                    parts = []
                    for r in si.findall('s:r', ns):
                        tr = r.find('s:t', ns)
                        if tr is not None:
                            parts.append(tr.text if tr.text else '')
                    strings.append(''.join(parts))
        except:
            strings = []
        
        # Get sheet data
        with z.open('xl/worksheets/sheet1.xml') as f:
            content = f.read().decode('utf-8')
        
        root = ET.fromstring(content)
        ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        
        # Parse rows
        for row in root.findall('.//s:row', ns):
            row_num = int(row.get('r'))
            cells = {}
            for cell in row.findall('s:c', ns):
                ref = cell.get('r')
                cell_type = cell.get('t', 'n')
                
                if cell_type == 's':  # shared string
                    val = cell.get('v', '0')
                    try:
                        idx = int(val)
                        cells[ref] = strings[idx] if idx < len(strings) else ''
                    except:
                        cells[ref] = ''
                elif cell_type == 'inlineStr':
                    inline = cell.find('s:is', ns)
                    if inline is not None:
                        t = inline.find('s:t', ns)
                        if t is not None:
                            cells[ref] = t.text if t.text else ''
                        else:
                            cells[ref] = ''
                    else:
                        cells[ref] = ''
                else:
                    val = cell.get('v')
                    if val is not None:
                        try:
                            cells[ref] = float(val)
                        except:
                            cells[ref] = val
                    else:
                        cells[ref] = ''
            
            data[row_num] = cells
    
    return data

def load_descriptors(filepath):
    """Load molecular descriptor data from Excel file"""
    if HAS_PANDAS:
        df = pd.read_excel(filepath)
        return df
    else:
        data = excel_to_dataframe_nopandas(filepath)
        return data

def normalize_descriptors(descriptors_matrix):
    """
    Normalize each descriptor to [0, 1] range using formula:
    vn = (v - min(ld)) / (max(ld) - min(ld))
    """
    # Handle missing values (-999 or NaN)
    descriptors_matrix = np.array(descriptors_matrix, dtype=float)
    descriptors_matrix[descriptors_matrix == -999] = np.nan
    
    # Get min and max for each column (descriptor)
    col_min = np.nanmin(descriptors_matrix, axis=0)
    col_max = np.nanmax(descriptors_matrix, axis=0)
    
    # Avoid division by zero
    col_range = col_max - col_min
    col_range[col_range == 0] = 1
    
    # Normalize
    normalized = (descriptors_matrix - col_min) / col_range
    
    # Replace NaN with 0
    normalized = np.nan_to_num(normalized, nan=0.0)
    
    return normalized

def create_mixture_vector(component_indices, descriptor_matrix):
    """
    Create mixture vector by summing component descriptors
    Then normalize by dividing by the norm (to eliminate effect of number of components)
    """
    # Sum component vectors
    mixture_vector = np.sum(descriptor_matrix[component_indices], axis=0)
    
    # Normalize by dividing by its norm
    norm = np.linalg.norm(mixture_vector)
    if norm > 0:
        mixture_vector = mixture_vector / norm
    
    return mixture_vector

def vector_angle(vector1, vector2):
    """
    Calculate angle between two vectors using dot product formula:
    angle = arccos((U·V) / (|U||V|))
    
    Returns angle in degrees
    """
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    
    # Handle edge cases
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Clamp to avoid numerical errors
    cos_angle = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
    
    # Return angle in degrees
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

def information_entropy(vector):
    """
    Calculate Shannon information entropy of a descriptor vector.
    Treats each descriptor value as a probability distribution after normalization.
    """
    # Normalize vector to sum to 1 (treat as probability distribution)
    abs_vector = np.abs(vector)
    total = np.sum(abs_vector)
    
    if total == 0:
        return 0.0
    
    probability = abs_vector / total
    
    # Calculate Shannon entropy: H = -sum(p * log2(p))
    # Handle zero probabilities (0 * log2(0) = 0)
    entropy = 0.0
    for p in probability:
        if p > 0:
            entropy -= p * np.log2(p)
    
    return entropy

def relative_perceptual_complexity(entropy, max_entropy=None):
    """
    Calculate relative perceptual complexity.
    Higher entropy means more complex/informative mixture.
    
    If max_entropy not provided, use log2 of descriptor count as maximum possible
    """
    if max_entropy is None:
        # Maximum possible entropy is log2(n) for n descriptors
        max_entropy = np.log2(len(entropy))
    
    # Relative complexity = entropy / max_entropy
    complexity = entropy / max_entropy if max_entropy > 0 else 0
    
    return complexity

def predict_perceptual_similarity(mixture_vector1, mixture_vector2):
    """
    Predict perceptual similarity between two mixtures.
    Smaller angle = higher similarity.
    Returns angle and similarity score (0-100 scale)
    """
    angle = vector_angle(mixture_vector1, mixture_vector2)
    
    # Convert angle to similarity score (0-100 scale)
    # Angle 0 -> similarity 100, Angle 180 -> similarity 0
    similarity = ((180 - angle) / 180) * 100
    
    return angle, similarity

def main():
    print("=" * 70)
    print("Odorant Mixture Perceptual Similarity Prediction")
    print("Based on Snitz et al. (2013) PLoS Comput Biology")
    print("=" * 70)
    
    # File paths
    descriptor_file = '13-descriptors-dragon.xlsx'
    mixture_file = 'mixture-components.xlsx'
    
    print("\n[1] Loading molecular descriptors...")
    try:
        if HAS_PANDAS:
            desc_df = pd.read_excel(descriptor_file)
            print(f"   Loaded {desc_df.shape[0]} molecules, {desc_df.shape[1]} descriptors")
            print(f"   Columns: {desc_df.columns.tolist()[:10]}...")
            descriptor_matrix = desc_df.values
            molecule_ids = desc_df['MOL_ID'].values if 'MOL_ID' in desc_df.columns else None
        else:
            print("   Warning: pandas not available, using basic parsing")
            return
    except Exception as e:
        print(f"Error loading descriptors: {e}")
        return
    
    print("\n[2] Loading mixture composition data...")
    try:
        if HAS_PANDAS:
            mix_df = pd.read_excel(mixture_file)
            print(f"   Mixture data shape: {mix_df.shape}")
            print(mix_df)
    except Exception as e:
        print(f"Error loading mixture data: {e}")
        return
    
    print("\n[3] Normalizing descriptors to [0,1] range...")
    normalized_descriptors = normalize_descriptors(descriptor_matrix[:, 1:])  # Skip first column (MOL_ID)
    print(f"   Normalized matrix shape: {normalized_descriptors.shape}")
    
    print("\n[4] Example calculations:")
    
    # Example: If we have 2 mixtures (e.g., first two rows of mixture data)
    # Assuming mixture components are specified by column indices
    print("\n   Example: Calculating similarity between two mixtures")
    
    # Example: Mix 1 (e.g., High similarity group) vs Mix 2
    # This is a placeholder - actual calculation depends on mixture composition
    # Assuming mixture_data contains indices of components
    
    # Example with first 2 different mixtures from the data
    # (Replace with actual component indices from your data)
    
    # Demo with some sample molecules
    n_molecules = min(5, normalized_descriptors.shape[0])
    
    print(f"\n   Using first {n_molecules} molecules as sample components")
    
    # Example mixture 1: first 3 molecules
    mix1_indices = list(range(3))
    # Example mixture 2: molecules 1, 2, 4 (skip one)
    mix2_indices = [0, 1, 3] if n_molecules > 3 else [0, 1, 2]
    
    mix1_vector = create_mixture_vector(mix1_indices, normalized_descriptors)
    mix2_vector = create_mixture_vector(mix2_indices, normalized_descriptors)
    
    print(f"\n   Mixture 1 components: {mix1_indices}")
    print(f"   Mixture 2 components: {mix2_indices}")
    
    # Calculate angle and similarity
    angle, similarity = predict_perceptual_similarity(mix1_vector, mix2_vector)
    print(f"\n   Vector angle: {angle:.2f} degrees")
    print(f"   Predicted perceptual similarity: {similarity:.2f}/100")
    
    # Calculate entropy for each mixture
    entropy1 = information_entropy(mix1_vector)
    entropy2 = information_entropy(mix2_vector)
    
    print(f"\n   Mixture 1 information entropy: {entropy1:.4f}")
    print(f"   Mixture 2 information entropy: {entropy2:.4f}")
    
    # Calculate relative perceptual complexity
    complexity1 = relative_perceptual_complexity(entropy1)
    complexity2 = relative_perceptual_complexity(entropy2)
    
    print(f"\n   Mixture 1 relative perceptual complexity: {complexity1:.4f}")
    print(f"   Mixture 2 relative perceptual complexity: {complexity2:.4f}")
    
    print("\n" + "=" * 70)
    print("USAGE GUIDE:")
    print("=" * 70)
    print("""
This program predicts perceptual similarity between odorant mixtures.

Key functions:
1. normalize_descriptors() - Normalizes Dragon descriptors to [0,1]
2. create_mixture_vector() - Creates total mixture vector from components  
3. vector_angle() - Calculates angle between two mixture vectors
4. information_entropy() - Calculates Shannon entropy of mixture descriptors
5. relative_perceptual_complexity() - Calculates relative complexity
6. predict_perceptual_similarity() - Main prediction function

To use with your data:
1. Ensure '13-descriptors-dragon.xlsx' has molecule IDs and their Dragon descriptors
2. Ensure 'mixture-components.xlsx' specifies mixture compositions
3. Modify the mixture_indices in the code to match your data
4. Run to get similarity predictions, entropy, and complexity values

Output interpretation:
- Smaller angle → higher perceptual similarity
- Higher entropy → more complex/informative mixture
- Relative complexity ranges from 0 to 1
""")

if __name__ == "__main__":
    main()