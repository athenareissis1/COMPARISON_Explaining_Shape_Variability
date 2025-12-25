#!/usr/bin/env python3
"""
ADNI Hippocampus Batch Rigid Registration

This script performs rigid registration (alignment) on batch-processed PLY files.
Rigid registration applies only rotation and translation - NO scaling or deformation.

PRESERVES:
✓ Volumes (rigid transformations are volume-preserving)
✓ All distances within each mesh
✓ Shape geometry and topology
✓ Previous volume corrections

CHANGES:
✗ Absolute coordinates (aligned to reference medoid)
✗ Pose/orientation (that's the purpose - removes pose variation)

Process:
1. Load all PLY files from each group (left, right, combined)
2. Find reference medoid shape for each group
3. Apply rigid registration (rotation + translation) to align all to medoid
4. Save aligned PLY files to new directories with _rigid_reg suffix

Input Structure:
  adni_processed/
    left_hippocampus_ply/
      minimal/
      minimal_smooth/
      metadata.csv
    right_hippocampus_ply/
      minimal/
      minimal_smooth/
      metadata.csv
    combined_hippocampus_ply/
      minimal/
      minimal_smooth/
      metadata.csv

Output Structure:
  adni_processed/
    left_hippocampus_ply_rigid_reg/
      minimal/               - Rigidly aligned minimal PLY files
      minimal_smooth/        - Rigidly aligned smooth PLY files
      metadata.csv          - Original metadata + registration info
      reference_medoid.txt  - Info about reference shape
    right_hippocampus_ply_rigid_reg/
      minimal/
      minimal_smooth/
      metadata.csv
      reference_medoid.txt
    combined_hippocampus_ply_rigid_reg/
      minimal/
      minimal_smooth/
      metadata.csv
      reference_medoid.txt

Requirements:
- Batch-processed PLY files must exist (from batch_process_to_ply.py)
- ShapeWorks library for rigid registration

Usage:
    python batch_rigid_registration.py
"""

import os
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import shapeworks as sw
import shutil
from contextlib import redirect_stdout, redirect_stderr, contextmanager, ExitStack


@contextmanager
def suppress_stdout(out=True, err=False):
    """Suppress stdout and/or stderr output to reduce console clutter"""
    with ExitStack() as stack:
        with open(os.devnull, "w") as null:
            if out:
                stack.enter_context(redirect_stdout(null))
            if err:
                stack.enter_context(redirect_stderr(null))
            yield


def rigid_register_group(input_ply_dir, output_ply_dir, group_name, version_name):
    """
    Perform rigid registration on a group of PLY files.
    
    Steps:
    1. Load all PLY files as ShapeWorks meshes
    2. Find reference medoid shape (most representative)
    3. Compute rigid transformation for each mesh to align with medoid
    4. Apply transformation and save aligned meshes
    
    Args:
        input_ply_dir: Directory containing input PLY files
        output_ply_dir: Directory to save aligned PLY files
        group_name: Name of group (left, right, combined)
        version_name: Version name (minimal or minimal_smooth)
    
    Returns:
        dict: Registration statistics and reference info
    """
    print(f"\n  Processing {version_name}...")
    
    # Get all PLY files
    ply_files = sorted(glob(os.path.join(input_ply_dir, "*.ply")))
    
    if len(ply_files) == 0:
        print(f"    ⚠ No PLY files found in {input_ply_dir}")
        return None
    
    print(f"    Found {len(ply_files)} PLY files")
    
    # Load all meshes
    print(f"    Loading meshes...")
    mesh_dict = {}
    failed_loads = 0
    
    for ply_file in tqdm(ply_files, desc=f"    Loading", leave=False):
        try:
            with suppress_stdout():
                basename = os.path.basename(ply_file).replace('.ply', '')
                mesh = sw.Mesh(ply_file)
                mesh_dict[basename] = mesh
        except Exception as e:
            print(f"    ✗ Failed to load {basename}: {e}")
            failed_loads += 1
    
    if len(mesh_dict) == 0:
        print(f"    ✗ Failed to load any meshes!")
        return None
    
    print(f"    ✓ Loaded {len(mesh_dict)}/{len(ply_files)} meshes")
    if failed_loads > 0:
        print(f"    ⚠ Failed to load {failed_loads} meshes")
    
    # Find reference medoid shape
    print(f"    Finding reference medoid shape...")
    mesh_list = list(mesh_dict.values())
    
    with suppress_stdout():
        ref_index = sw.find_reference_mesh_index(mesh_list)
    
    ref_mesh = mesh_list[ref_index]
    ref_name = list(mesh_dict.keys())[ref_index]
    
    print(f"    ✓ Reference medoid: {ref_name}")
    
    # Create output directory
    os.makedirs(output_ply_dir, exist_ok=True)
    
    # Perform rigid registration for all meshes
    print(f"    Applying rigid registration...")
    success_count = 0
    failed_count = 0
    
    for name, mesh in tqdm(mesh_dict.items(), desc=f"    Registering", leave=False):
        try:
            with suppress_stdout():
                # Compute rigid transformation to align with reference
                # AlignmentType.Rigid: rotation + translation only (no scaling)
                # 100 iterations for convergence
                rigid_transform = mesh.createTransform(
                    ref_mesh, 
                    sw.Mesh.AlignmentType.Rigid, 
                    100
                )
                
                # Apply rigid transformation
                mesh.applyTransform(rigid_transform)
                
                # Save aligned mesh
                output_file = os.path.join(output_ply_dir, f"{name}.ply")
                mesh.write(output_file)
                
                success_count += 1
        except Exception as e:
            print(f"    ✗ Failed to register {name}: {e}")
            failed_count += 1
    
    print(f"    ✓ Registration complete:")
    print(f"      Success: {success_count}/{len(mesh_dict)}")
    if failed_count > 0:
        print(f"      Failed: {failed_count}")
    
    return {
        'total_files': len(ply_files),
        'loaded': len(mesh_dict),
        'failed_load': failed_loads,
        'registered': success_count,
        'failed_register': failed_count,
        'reference_medoid': ref_name,
        'reference_index': ref_index
    }


def process_group(input_base_dir, output_base_dir, group_name):
    """
    Process rigid registration for a group (left, right, or combined).
    
    This processes both minimal and minimal_smooth versions separately.
    Each version gets its own medoid and registration.
    
    Args:
        input_base_dir: Base directory with minimal/ and minimal_smooth/ subdirs
        output_base_dir: Base directory to save aligned PLY files
        group_name: Name of group (left, right, combined)
    
    Returns:
        dict: Processing statistics for this group
    """
    print(f"\n{'='*80}")
    print(f"RIGID REGISTRATION: {group_name.upper()} HIPPOCAMPUS")
    print(f"{'='*80}")
    
    # Check if input directories exist
    minimal_input = os.path.join(input_base_dir, "minimal")
    smooth_input = os.path.join(input_base_dir, "minimal_smooth")
    
    if not os.path.exists(minimal_input) and not os.path.exists(smooth_input):
        print(f"✗ Error: No input directories found in {input_base_dir}")
        return None
    
    # Create output directories
    minimal_output = os.path.join(output_base_dir, "minimal")
    smooth_output = os.path.join(output_base_dir, "minimal_smooth")
    os.makedirs(output_base_dir, exist_ok=True)
    
    results = {}
    
    # Process MINIMAL version
    if os.path.exists(minimal_input):
        print(f"\nStep 1: Processing MINIMAL version")
        minimal_result = rigid_register_group(
            minimal_input, 
            minimal_output, 
            group_name, 
            "minimal"
        )
        if minimal_result:
            results['minimal'] = minimal_result
    else:
        print(f"\n  ⚠ Skipping MINIMAL (directory not found)")
    
    # Process MINIMAL_SMOOTH version
    if os.path.exists(smooth_input):
        print(f"\nStep 2: Processing MINIMAL_SMOOTH version")
        smooth_result = rigid_register_group(
            smooth_input, 
            smooth_output, 
            group_name, 
            "minimal_smooth"
        )
        if smooth_result:
            results['minimal_smooth'] = smooth_result
    else:
        print(f"\n  ⚠ Skipping MINIMAL_SMOOTH (directory not found)")
    
    if len(results) == 0:
        print(f"\n✗ No successful processing for {group_name}")
        return None
    
    # Copy and update metadata from original processing
    print(f"\nStep 3: Updating metadata...")
    
    original_metadata_file = os.path.join(input_base_dir, "metadata.csv")
    output_metadata_file = os.path.join(output_base_dir, "metadata.csv")
    
    if os.path.exists(original_metadata_file):
        # Load original metadata
        metadata_df = pd.read_csv(original_metadata_file)
        
        # Add rigid registration information
        metadata_df['rigid_registered'] = True
        metadata_df['minimal_ref_medoid'] = results.get('minimal', {}).get('reference_medoid', None)
        metadata_df['minimal_registered'] = results.get('minimal', {}).get('registered', 0)
        metadata_df['smooth_ref_medoid'] = results.get('minimal_smooth', {}).get('reference_medoid', None)
        metadata_df['smooth_registered'] = results.get('minimal_smooth', {}).get('registered', 0)
        
        # Save updated metadata
        metadata_df.to_csv(output_metadata_file, index=False)
        print(f"  ✓ Metadata saved to: {output_metadata_file}")
    else:
        print(f"  ⚠ Original metadata not found, creating new metadata...")
        
        # Create new metadata
        metadata = {
            'group': [group_name],
            'rigid_registered': [True],
            'minimal_ref_medoid': [results.get('minimal', {}).get('reference_medoid', None)],
            'minimal_registered': [results.get('minimal', {}).get('registered', 0)],
            'minimal_failed': [results.get('minimal', {}).get('failed_register', 0)],
            'smooth_ref_medoid': [results.get('minimal_smooth', {}).get('reference_medoid', None)],
            'smooth_registered': [results.get('minimal_smooth', {}).get('registered', 0)],
            'smooth_failed': [results.get('minimal_smooth', {}).get('failed_register', 0)],
        }
        
        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(output_metadata_file, index=False)
        print(f"  ✓ New metadata saved to: {output_metadata_file}")
    
    # Save reference medoid information
    print(f"\nStep 4: Saving reference information...")
    
    ref_info_file = os.path.join(output_base_dir, "reference_medoid.txt")
    with open(ref_info_file, 'w') as f:
        f.write(f"{group_name.upper()} HIPPOCAMPUS - RIGID REGISTRATION\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"REFERENCE MEDOID INFORMATION\n")
        f.write(f"{'='*80}\n\n")
        
        if 'minimal' in results:
            f.write(f"MINIMAL version:\n")
            f.write(f"  Reference medoid: {results['minimal']['reference_medoid']}\n")
            f.write(f"  Reference index: {results['minimal']['reference_index']}\n")
            f.write(f"  Registered meshes: {results['minimal']['registered']}/{results['minimal']['loaded']}\n\n")
        
        if 'minimal_smooth' in results:
            f.write(f"MINIMAL_SMOOTH version:\n")
            f.write(f"  Reference medoid: {results['minimal_smooth']['reference_medoid']}\n")
            f.write(f"  Reference index: {results['minimal_smooth']['reference_index']}\n")
            f.write(f"  Registered meshes: {results['minimal_smooth']['registered']}/{results['minimal_smooth']['loaded']}\n\n")
        
        f.write(f"\nRIGID REGISTRATION PROPERTIES\n")
        f.write(f"{'='*80}\n")
        f.write(f"Rigid registration preserves:\n")
        f.write(f"  ✓ Volumes (no scaling applied)\n")
        f.write(f"  ✓ All distances within each mesh\n")
        f.write(f"  ✓ Shape geometry and topology\n")
        f.write(f"  ✓ Previous volume corrections\n\n")
        f.write(f"Rigid registration changes:\n")
        f.write(f"  • Absolute coordinates (aligned to reference)\n")
        f.write(f"  • Pose/orientation (removes pose variation)\n\n")
        f.write(f"USAGE\n")
        f.write(f"{'='*80}\n")
        f.write(f"These aligned meshes are suitable for:\n")
        f.write(f"  • Deformetrica dense correspondence registration\n")
        f.write(f"  • Direct mesh comparison (Chamfer distance, etc.)\n")
        f.write(f"  • Statistical shape analysis\n")
        f.write(f"  • Visualization (all in same coordinate frame)\n\n")
        f.write(f"Volume measurements remain accurate:\n")
        f.write(f"  • Use volume_unscale_factor from original metadata.csv\n")
        f.write(f"  • original_volume_mm3 = ply_volume × volume_unscale_factor\n")
    
    print(f"  ✓ Reference info saved to: {ref_info_file}")
    
    return {
        'group': group_name,
        'minimal': results.get('minimal'),
        'minimal_smooth': results.get('minimal_smooth'),
        'output_dir': output_base_dir
    }


def main():
    """Main batch rigid registration function"""
    
    print("="*80)
    print("ADNI HIPPOCAMPUS BATCH RIGID REGISTRATION")
    print("Volume-Preserving Alignment to Reference Medoid")
    print("="*80)
    
    # Set paths
    base_dir = "/home/jakaria/ADNI/ADNI_1/adni_processed/"
    
    # Input directories (batch-processed PLY files)
    left_input = os.path.join(base_dir, "left_hippocampus_ply")
    right_input = os.path.join(base_dir, "right_hippocampus_ply")
    combined_input = os.path.join(base_dir, "combined_hippocampus_ply")
    
    # Output directories (rigidly registered PLY files)
    left_output = os.path.join(base_dir, "left_hippocampus_ply_rigid_reg")
    right_output = os.path.join(base_dir, "right_hippocampus_ply_rigid_reg")
    combined_output = os.path.join(base_dir, "combined_hippocampus_ply_rigid_reg")
    
    # Check input directories
    print(f"\nChecking input directories...")
    groups_to_process = []
    
    if os.path.exists(left_input):
        print(f"  ✓ Left hippocampus PLY files found")
        groups_to_process.append(('left', left_input, left_output))
    else:
        print(f"  ✗ Left hippocampus PLY files NOT found")
    
    if os.path.exists(right_input):
        print(f"  ✓ Right hippocampus PLY files found")
        groups_to_process.append(('right', right_input, right_output))
    else:
        print(f"  ✗ Right hippocampus PLY files NOT found")
    
    if os.path.exists(combined_input):
        print(f"  ✓ Combined hippocampus PLY files found")
        groups_to_process.append(('combined', combined_input, combined_output))
    else:
        print(f"  ✗ Combined hippocampus PLY files NOT found")
    
    if len(groups_to_process) == 0:
        print(f"\n❌ ERROR: No batch-processed PLY files found!")
        print(f"Please run batch_process_to_ply.py first.")
        return
    
    print(f"\nFound {len(groups_to_process)} group(s) to process")
    
    # Process each group
    results = []
    
    for group_name, input_dir, output_dir in groups_to_process:
        result = process_group(input_dir, output_dir, group_name)
        if result:
            results.append(result)
    
    # ============================================================================
    # Final Summary
    # ============================================================================
    print(f"\n{'='*80}")
    print("BATCH RIGID REGISTRATION COMPLETE - SUMMARY")
    print(f"{'='*80}")
    
    for result in results:
        print(f"\n{result['group'].upper()} HIPPOCAMPUS:")
        
        if result.get('minimal'):
            minimal = result['minimal']
            print(f"  MINIMAL version:")
            print(f"    Files: {minimal['total_files']}")
            print(f"    Registered: {minimal['registered']}")
            print(f"    Reference medoid: {minimal['reference_medoid']}")
        
        if result.get('minimal_smooth'):
            smooth = result['minimal_smooth']
            print(f"  MINIMAL_SMOOTH version:")
            print(f"    Files: {smooth['total_files']}")
            print(f"    Registered: {smooth['registered']}")
            print(f"    Reference medoid: {smooth['reference_medoid']}")
        
        print(f"  Output directory: {result['output_dir']}")
    
    print(f"\n{'='*80}")
    print("KEY POINTS")
    print(f"{'='*80}")
    print(f"✓ Rigid registration applied (rotation + translation only)")
    print(f"✓ PRESERVES: Volumes, distances, shape geometry")
    print(f"✓ CHANGES: Absolute coordinates, pose/orientation")
    print(f"✓ Each version (minimal/smooth) has its own reference medoid")
    print(f"✓ All meshes aligned to their respective medoid shapes")
    print(f"✓ Metadata and reference info saved in each output directory")
    
    print(f"\n{'='*80}")
    print("OUTPUT STRUCTURE")
    print(f"{'='*80}")
    for result in results:
        print(f"\n{result['output_dir']}/")
        print(f"  ├── minimal/             (rigidly aligned minimal PLY files)")
        print(f"  ├── minimal_smooth/      (rigidly aligned smooth PLY files)")
        print(f"  ├── metadata.csv         (original metadata + registration info)")
        print(f"  └── reference_medoid.txt (reference shape information)")
    
    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print(f"1. Use these aligned meshes for:")
    print(f"   • Deformetrica dense correspondence registration")
    print(f"   • Direct mesh comparison (Chamfer/EMD distances)")
    print(f"   • Statistical shape analysis")
    print(f"   • Visualization (all in same coordinate frame)")
    print(f"\n2. Volume measurements remain accurate:")
    print(f"   • Use volume_unscale_factor from metadata.csv")
    print(f"   • Rigid transformations preserve volumes")
    print(f"\n3. Check reference_medoid.txt to see which shape was used as reference")


if __name__ == "__main__":
    main()
