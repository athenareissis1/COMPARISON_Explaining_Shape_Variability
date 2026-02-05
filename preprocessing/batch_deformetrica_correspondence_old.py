#!/usr/bin/env python3
"""
ADNI Hippocampus Dense Correspondence with Volume Preservation

This script creates dense point correspondence while preserving original volumes:
1. Converts rigidly-aligned PLY to VTK (for Deformetrica)
2. Runs Deformetrica atlas estimation to establish correspondence
3. Rescales reconstructed meshes to match original volumes
4. Validates volume accuracy
5. Saves final meshes with correspondence AND correct volumes

Usage:
    python batch_deformetrica_correspondence.py --group left --version minimal
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import trimesh
from glob import glob
from tqdm import tqdm
from pathlib import Path
import deformetrica as dfca


def ply_to_vtk(ply_file, vtk_file):
    """Convert PLY to VTK format using trimesh"""
    mesh = trimesh.load(ply_file)
    mesh.export(vtk_file, file_type='vtk')


def calculate_mesh_volume(mesh_file):
    """Calculate mesh volume using trimesh"""
    mesh = trimesh.load(mesh_file)
    return abs(mesh.volume)


def rescale_mesh_to_volume(input_mesh, output_mesh, target_volume):
    """Rescale mesh uniformly to match target volume"""
    mesh = trimesh.load(input_mesh)
    current_volume = abs(mesh.volume)
    
    # Calculate uniform scale factor: s = (V_target / V_current)^(1/3)
    scale_factor = (target_volume / current_volume) ** (1.0 / 3.0)
    
    # Apply uniform scaling (preserves correspondence)
    mesh.vertices *= scale_factor
    
    # Save rescaled mesh
    mesh.export(output_mesh)
    
    return scale_factor


def run_deformetrica_atlas(vtk_dir, template_file, output_dir):
    """Run Deformetrica atlas estimation"""
    
    # Get all VTK files
    vtk_files = sorted(glob(os.path.join(vtk_dir, "*.vtk")))
    
    if len(vtk_files) == 0:
        raise ValueError(f"No VTK files found in {vtk_dir}")
    
    print(f"  Found {len(vtk_files)} VTK files for Deformetrica")
    
    # Instantiate Deformetrica
    deformetrica = dfca.Deformetrica(output_dir=str(output_dir), verbosity="INFO")
    
    # Prepare dataset
    dataset_filenames = []
    subject_ids = []
    for vtk_file in vtk_files:
        subject_id = Path(vtk_file).stem
        dataset_filenames.append([{"hippo": str(vtk_file)}])
        subject_ids.append(subject_id)
    
    dataset_specifications = {
        "dataset_filenames": dataset_filenames,
        "subject_ids": subject_ids,
    }
    
    # Template specifications
    template_specifications = {
        "hippo": {
            "deformable_object_type": "SurfaceMesh",
            "kernel_type": "keops",
            "kernel_width": 0.03,
            "noise_std": 0.01,
            "filename": str(template_file),
            "attachment_type": "varifold",
        }
    }
    
    # Estimator options
    estimator_options = {
        "optimization_method_type": "GradientAscent",
        "max_line_search_iterations": 10,
        "gpu_mode": "auto",
        "max_iterations": 100,
        "initial_step_size": 0.5,
        "convergence_tolerance": 1e-6,
        "save_every_n_iters": 20,
    }
    
    # Model options
    model_options = {
        "deformation_kernel_type": "keops",
        "deformation_kernel_width": 0.05,
        "number_of_timepoints": 25,
    }
    
    # Run atlas estimation
    print("  Running Deformetrica atlas estimation (this may take a while)...")
    model = deformetrica.estimate_deterministic_atlas(
        template_specifications,
        dataset_specifications,
        estimator_options=estimator_options,
        model_options=model_options,
    )
    
    return model, subject_ids


def load_original_volumes(metadata_file, volume_unscale_factor):
    """Load original volumes from rigidly-aligned PLY files"""
    # Original volumes stored as: original_volume_mm3 = ply_volume * volume_unscale_factor
    # We'll calculate PLY volumes directly from the rigidly-aligned files
    return volume_unscale_factor


def process_group(group_name, version_name, base_dir):
    """Process one group (left/right/combined) with one version (minimal/smooth)"""
    
    print(f"\n{'='*80}")
    print(f"DEFORMETRICA CORRESPONDENCE: {group_name.upper()} - {version_name.upper()}")
    print(f"{'='*80}")
    
    # Paths
    rigid_reg_dir = os.path.join(base_dir, f"{group_name}_hippocampus_ply_rigid_reg")
    ply_input_dir = os.path.join(rigid_reg_dir, version_name)
    metadata_file = os.path.join(rigid_reg_dir, "metadata.csv")
    
    output_base = os.path.join(base_dir, f"{group_name}_hippocampus_correspondence")
    vtk_dir = os.path.join(output_base, f"{version_name}_vtk")
    deformetrica_output = os.path.join(output_base, f"{version_name}_deformetrica")
    final_output = os.path.join(output_base, f"{version_name}_final")
    
    os.makedirs(vtk_dir, exist_ok=True)
    os.makedirs(deformetrica_output, exist_ok=True)
    os.makedirs(final_output, exist_ok=True)
    
    # Check inputs
    if not os.path.exists(ply_input_dir):
        print(f"✗ Error: Input directory not found: {ply_input_dir}")
        return None
    
    if not os.path.exists(metadata_file):
        print(f"✗ Error: Metadata file not found: {metadata_file}")
        return None
    
    # Load metadata for volume unscale factor
    metadata_df = pd.read_csv(metadata_file)
    volume_unscale = metadata_df['volume_unscale_factor'].values[0]
    
    print(f"\nStep 1: Converting PLY to VTK")
    ply_files = sorted(glob(os.path.join(ply_input_dir, "*.ply")))
    print(f"  Found {len(ply_files)} PLY files")
    
    # Store original volumes (from rigidly-aligned PLY files)
    original_volumes = {}
    
    for ply_file in tqdm(ply_files, desc="  Converting"):
        basename = os.path.basename(ply_file).replace('.ply', '')
        vtk_file = os.path.join(vtk_dir, f"{basename}.vtk")
        ply_to_vtk(ply_file, vtk_file)
        
        # Calculate and store original volume in mm³
        ply_volume = calculate_mesh_volume(ply_file)
        original_volumes[basename] = ply_volume * volume_unscale
    
    print(f"  ✓ Converted {len(ply_files)} files to VTK")
    
    # Use first mesh as template (medoid would be better, but this is simpler)
    template_file = os.path.join(vtk_dir, os.listdir(vtk_dir)[0])
    
    print(f"\nStep 2: Running Deformetrica Atlas Estimation")
    model, subject_ids = run_deformetrica_atlas(vtk_dir, template_file, deformetrica_output)
    print(f"  ✓ Deformetrica completed")
    
    # Find reconstructed meshes
    reconstruction_pattern = os.path.join(deformetrica_output, "DeterministicAtlas__Reconstruction__*.vtk")
    reconstructed_files = sorted(glob(reconstruction_pattern))
    
    if len(reconstructed_files) == 0:
        print(f"✗ Error: No reconstructed meshes found")
        return None
    
    print(f"\nStep 3: Rescaling Meshes to Original Volumes")
    print(f"  Found {len(reconstructed_files)} reconstructed meshes")
    
    volume_errors = []
    success_count = 0
    
    for recon_file in tqdm(reconstructed_files, desc="  Rescaling"):
        # Extract subject ID from filename
        # Format: DeterministicAtlas__Reconstruction__subject_id__hippo.vtk
        basename = os.path.basename(recon_file)
        subject_id = basename.replace("DeterministicAtlas__Reconstruction__", "").replace("__hippo.vtk", "")
        
        if subject_id not in original_volumes:
            print(f"  ⚠ Warning: Subject {subject_id} not found in original volumes")
            continue
        
        target_volume = original_volumes[subject_id]
        final_file = os.path.join(final_output, f"{subject_id}.vtk")
        
        # Rescale to match original volume
        scale_factor = rescale_mesh_to_volume(recon_file, final_file, target_volume)
        
        # Validate final volume
        final_volume = calculate_mesh_volume(final_file) * volume_unscale
        volume_error = abs(final_volume - target_volume) / target_volume * 100
        volume_errors.append(volume_error)
        
        success_count += 1
    
    print(f"  ✓ Rescaled {success_count}/{len(reconstructed_files)} meshes")
    
    # Volume validation statistics
    print(f"\nStep 4: Volume Validation")
    volume_errors = np.array(volume_errors)
    print(f"  Volume error statistics (%):")
    print(f"    Mean: {np.mean(volume_errors):.4f}%")
    print(f"    Std: {np.std(volume_errors):.4f}%")
    print(f"    Max: {np.max(volume_errors):.4f}%")
    print(f"    Min: {np.min(volume_errors):.4f}%")
    
    # Check if errors are acceptable (<1% is excellent)
    if np.mean(volume_errors) < 1.0:
        print(f"  ✓ PASS: Volume errors are excellent (mean < 1%)")
    elif np.mean(volume_errors) < 5.0:
        print(f"  ✓ PASS: Volume errors are acceptable (mean < 5%)")
    else:
        print(f"  ⚠ WARNING: Volume errors are high (mean > 5%)")
    
    # Save summary
    summary = {
        'group': group_name,
        'version': version_name,
        'total_meshes': len(reconstructed_files),
        'successful': success_count,
        'volume_error_mean': np.mean(volume_errors),
        'volume_error_std': np.std(volume_errors),
        'volume_error_max': np.max(volume_errors),
        'volume_error_min': np.min(volume_errors),
        'volume_unscale_factor': volume_unscale,
    }
    
    summary_df = pd.DataFrame([summary])
    summary_file = os.path.join(output_base, f"{version_name}_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\n  ✓ Summary saved to: {summary_file}")
    
    # Save README
    readme_file = os.path.join(output_base, f"{version_name}_README.txt")
    with open(readme_file, 'w') as f:
        f.write(f"{group_name.upper()} {version_name.upper()} - CORRESPONDENCE WITH VOLUME PRESERVATION\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"MESH PROPERTIES:\n")
        f.write(f"  • Dense point correspondence: All meshes have identical topology\n")
        f.write(f"  • Vertex i corresponds to vertex i across all subjects\n")
        f.write(f"  • Volumes match original (within {np.mean(volume_errors):.2f}% error)\n\n")
        f.write(f"VOLUME VALIDATION:\n")
        f.write(f"  • Mean error: {np.mean(volume_errors):.4f}%\n")
        f.write(f"  • Std error: {np.std(volume_errors):.4f}%\n")
        f.write(f"  • Max error: {np.max(volume_errors):.4f}%\n")
        f.write(f"  • Min error: {np.min(volume_errors):.4f}%\n\n")
        f.write(f"USAGE:\n")
        f.write(f"  • Use {version_name}_final/*.vtk for SpiralNet or mesh neural networks\n")
        f.write(f"  • All meshes have same vertex count and connectivity\n")
        f.write(f"  • Volumes are preserved for accurate biomarker analysis\n\n")
        f.write(f"DIRECTORIES:\n")
        f.write(f"  • {version_name}_vtk/: Converted rigidly-aligned meshes (input to Deformetrica)\n")
        f.write(f"  • {version_name}_deformetrica/: Raw Deformetrica output (correspondence, wrong volumes)\n")
        f.write(f"  • {version_name}_final/: Final meshes (correspondence + correct volumes)\n")
    
    print(f"  ✓ README saved to: {readme_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Dense Correspondence with Volume Preservation')
    parser.add_argument('--group', type=str, required=True, 
                       choices=['left', 'right', 'combined'],
                       help='Hippocampus group to process')
    parser.add_argument('--version', type=str, required=True,
                       choices=['minimal', 'minimal_smooth'],
                       help='Processing version to use')
    parser.add_argument('--base_dir', type=str, 
                       default='/home/jakaria/ADNI/ADNI_1/adni_processed/',
                       help='Base directory containing processed data')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ADNI HIPPOCAMPUS DENSE CORRESPONDENCE WITH VOLUME PRESERVATION")
    print("="*80)
    
    # Process the specified group and version
    result = process_group(args.group, args.version, args.base_dir)
    
    if result:
        print(f"\n{'='*80}")
        print("PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"\nGroup: {result['group']}")
        print(f"Version: {result['version']}")
        print(f"Successful meshes: {result['successful']}/{result['total_meshes']}")
        print(f"Volume error: {result['volume_error_mean']:.4f}% ± {result['volume_error_std']:.4f}%")
        print(f"\nFinal meshes ready for SpiralNet!")
    else:
        print(f"\n{'='*80}")
        print("PROCESSING FAILED")
        print(f"{'='*80}")
        sys.exit(1)


if __name__ == "__main__":
    main()
