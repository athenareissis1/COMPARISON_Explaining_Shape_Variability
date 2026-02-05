#!/usr/bin/env python3
"""
ADNI Hippocampus Dense Correspondence with Volume Preservation

This script creates dense point correspondence while preserving original volumes:
1. Converts rigidly-aligned PLY to VTK (for Deformetrica)
2. Runs Deformetrica atlas estimation to establish correspondence
3. Rescales reconstructed meshes to match original volumes
4. Saves final meshes with correspondence AND correct volumes (VTK and PLY)

Usage:
    python batch_deformetrica_correspondence.py --group left --version minimal --iterations 50
    
Notes:
    - Iterations: 50 recommended for large datasets (balance speed/quality)
    - More samples = can use fewer iterations
    - Test script used 100 iterations for 5 samples
    - Minimum recommended: 30 iterations
"""

import os
import sys
import argparse
import re
import numpy as np
import pandas as pd
import trimesh
from glob import glob
from tqdm import tqdm
from pathlib import Path
import deformetrica as dfca


def ply_to_vtk(ply_file, vtk_file):
    """Convert PLY to VTK format (Legacy ASCII format)"""
    mesh = trimesh.load(ply_file)
    
    # Write VTK Legacy ASCII format manually
    with open(vtk_file, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("vtk output\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        
        # Write vertices
        f.write(f"POINTS {len(mesh.vertices)} float\n")
        for v in mesh.vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        
        # Write faces
        n_faces = len(mesh.faces)
        f.write(f"\nPOLYGONS {n_faces} {n_faces * 4}\n")
        for face in mesh.faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def calculate_mesh_volume(mesh_file):
    """Calculate mesh volume - handles both PLY and VTK formats"""
    if mesh_file.endswith('.ply'):
        mesh = trimesh.load(mesh_file)
        return abs(mesh.volume)
    
    # Parse VTK manually
    vertices = []
    faces = []
    reading_points = False
    reading_polygons = False
    n_points = 0
    
    with open(mesh_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('POINTS'):
                n_points = int(line.split()[1])
                reading_points = True
                reading_polygons = False
                continue
            elif line.startswith('POLYGONS'):
                reading_points = False
                reading_polygons = True
                continue
            elif line.startswith('CELL') or line.startswith('POINT_DATA') or line.startswith('METADATA'):
                reading_points = False
                reading_polygons = False
                continue
            
            if reading_points and len(vertices) < n_points:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except:
                        pass
            
            elif reading_polygons:
                parts = line.split()
                if len(parts) >= 4 and parts[0] == '3':
                    try:
                        faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
                    except:
                        pass
    
    if len(vertices) > 0 and len(faces) > 0:
        mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
        return abs(mesh.volume)
    else:
        raise ValueError(f"Failed to parse VTK file: {mesh_file}")


def extract_image_id(filename):
    """Extract Image ID from filename (e.g., I123456)"""
    match = re.search(r'I\d+', filename)
    return match.group(0) if match else None


def rescale_mesh_to_volume(input_mesh, output_mesh_vtk, output_mesh_ply, target_volume):
    """
    Rescale mesh uniformly to match target volume, output to both VTK and PLY
    
    Args:
        input_mesh: Input VTK file (Deformetrica output)
        output_mesh_vtk: Output VTK file path
        output_mesh_ply: Output PLY file path
        target_volume: Target volume to match
    
    Returns:
        scale_factor: Applied scale factor
        final_volume: Final mesh volume (in scaled space)
    """
    # Read VTK file manually
    vertices = []
    faces = []
    with open(input_mesh, 'r') as f:
        reading_points = False
        reading_polygons = False
        n_points = 0
        
        for line in f:
            line = line.strip()
            
            if line.startswith('POINTS'):
                n_points = int(line.split()[1])
                reading_points = True
                reading_polygons = False
                continue
            elif line.startswith('POLYGONS'):
                reading_points = False
                reading_polygons = True
                continue
            elif line.startswith('CELL') or line.startswith('POINT_DATA') or line.startswith('METADATA'):
                reading_points = False
                reading_polygons = False
                continue
            
            if reading_points and len(vertices) < n_points:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except:
                        pass
            
            elif reading_polygons:
                parts = line.split()
                if len(parts) >= 4 and parts[0] == '3':
                    try:
                        faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
                    except:
                        pass
    
    # Create mesh
    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
    current_volume = abs(mesh.volume)
    
    # Calculate uniform scale factor: s = (V_target / V_current)^(1/3)
    scale_factor = (target_volume / current_volume) ** (1.0 / 3.0)
    
    # Apply uniform scaling (preserves correspondence topology)
    mesh.vertices *= scale_factor
    final_volume = abs(mesh.volume)
    
    # Save as VTK (manual write)
    with open(output_mesh_vtk, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("vtk output\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        
        # Write vertices
        f.write(f"POINTS {len(mesh.vertices)} float\n")
        for v in mesh.vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        
        # Write faces
        n_faces = len(mesh.faces)
        f.write(f"\nPOLYGONS {n_faces} {n_faces * 4}\n")
        for face in mesh.faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    # Save as PLY
    mesh.export(output_mesh_ply)
    
    return scale_factor, final_volume


def run_deformetrica_atlas(vtk_dir, template_file, output_dir, max_iterations=20):
    """
    Run Deformetrica atlas estimation
    
    Args:
        vtk_dir: Directory containing VTK files
        template_file: Initial template mesh
        output_dir: Output directory for Deformetrica
        max_iterations: Maximum iterations (default 50, recommended for large datasets)
    
    Returns:
        model: Deformetrica model
        subject_ids_list: List of subject IDs
    """
    
    # Get all VTK files
    vtk_files = sorted(glob(os.path.join(vtk_dir, "*.vtk")))
    
    if len(vtk_files) == 0:
        raise ValueError(f"No VTK files found in {vtk_dir}")
    
    print(f"  Found {len(vtk_files)} VTK files for Deformetrica")
    print(f"  Using {max_iterations} iterations")
    
    # Instantiate Deformetrica
    deformetrica = dfca.Deformetrica(output_dir=str(output_dir), verbosity="ERROR")
    
    # Prepare dataset with Image IDs as subject IDs
    dataset_filenames = []
    subject_ids_list = []
    
    for vtk_file in vtk_files:
        basename = Path(vtk_file).stem
        # Extract Image ID for subject ID
        image_id = extract_image_id(basename)
        if not image_id:
            print(f"  Warning: Could not extract Image ID from {basename}, using full name")
            image_id = basename
        
        dataset_filenames.append([{"hippo": str(vtk_file)}])
        subject_ids_list.append(image_id)
    
    dataset_specifications = {
        "dataset_filenames": dataset_filenames,
        "subject_ids": subject_ids_list,
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
        "max_iterations": max_iterations,
        "initial_step_size": 0.5,
        "convergence_tolerance": 1e-6,
        "save_every_n_iters": max(10, max_iterations // 5),  # Save 5 checkpoints
    }
    
    # Model options
    model_options = {
        "deformation_kernel_type": "keops",
        "deformation_kernel_width": 0.05,
        "number_of_timepoints": 25,
        # Don't freeze template - allow it to be estimated/refined
        # "freeze_template": True,
    }
    
    # Run atlas estimation (suppress iteration output)
    print("  Running Deformetrica atlas estimation...")
    print(f"  (This will take a while with {len(vtk_files)} subjects)")
    print("  Progress: iterations running silently in background...")
    
    # Redirect stdout/stderr to suppress iteration printing
    import sys
    from io import StringIO
    
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    try:
        model = deformetrica.estimate_deterministic_atlas(
            template_specifications,
            dataset_specifications,
            estimator_options=estimator_options,
            model_options=model_options,
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    # Find the learned template mesh (automatically saved by Deformetrica)
    template_output = os.path.join(output_dir, "DeterministicAtlas__EstimatedParameters__Template_hippo.vtk")
    if os.path.exists(template_output):
        print(f"  ✓ Template mesh found: {os.path.basename(template_output)}")
    else:
        print(f"  ⚠ Warning: Template mesh not found at expected location")
        template_output = None
    
    return model, subject_ids_list, template_output


def process_group(group_name, version_name, base_dir, max_iterations=50):
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
    final_output_vtk = os.path.join(output_base, f"{version_name}_final_vtk")
    final_output_ply = os.path.join(output_base, f"{version_name}_final_ply")
    
    os.makedirs(vtk_dir, exist_ok=True)
    os.makedirs(deformetrica_output, exist_ok=True)
    os.makedirs(final_output_vtk, exist_ok=True)
    os.makedirs(final_output_ply, exist_ok=True)
    
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
    
    print(f"\nMetadata Info:")
    print(f"  Volume unscale factor: {volume_unscale:.6e}")
    print(f"  This converts PLY volumes to mm³")
    
    print(f"\nStep 1: Converting PLY to VTK + Loading Original Volumes")
    ply_files = sorted(glob(os.path.join(ply_input_dir, "*.ply")))
    print(f"  Found {len(ply_files)} PLY files")
    
    # Store original volumes (in scaled space for matching with Deformetrica output)
    original_volumes = {}
    original_filenames = {}
    
    for ply_file in tqdm(ply_files, desc="  Processing"):
        basename = os.path.basename(ply_file).replace('.ply', '')
        
        # Extract Image ID for consistent naming
        image_id = extract_image_id(basename)
        if not image_id:
            print(f"  Warning: Could not extract Image ID from {basename}")
            continue
        
        # Store original filename
        original_filenames[image_id] = os.path.basename(ply_file)
        
        # Convert to VTK (named with Image ID for Deformetrica)
        vtk_file = os.path.join(vtk_dir, f"{image_id}.vtk")
        ply_to_vtk(ply_file, vtk_file)
        
        # Calculate and store original volume (in SCALED space)
        ply_volume = calculate_mesh_volume(ply_file)
        original_volumes[image_id] = ply_volume
    
    print(f"  ✓ Converted {len(original_volumes)} files to VTK")
    print(f"  Sample Image IDs: {list(original_volumes.keys())[:3]}")
    
    # Get medoid from rigid registration metadata (already computed)
    print(f"\nStep 2: Loading Reference Medoid from Rigid Registration")
    medoid_column = f"{version_name.replace('_smooth', '')}_ref_medoid"
    if version_name == 'minimal_smooth':
        medoid_column = 'smooth_ref_medoid'
    
    if medoid_column in metadata_df.columns:
        medoid_basename = metadata_df[medoid_column].values[0]
        if pd.notna(medoid_basename):
            # Extract Image ID from medoid basename
            ref_image_id = extract_image_id(medoid_basename)
            if ref_image_id:
                template_file = os.path.join(vtk_dir, f"{ref_image_id}.vtk")
                if os.path.exists(template_file):
                    print(f"  ✓ Using saved medoid: {ref_image_id}")
                    print(f"  Template file: {os.path.basename(template_file)}")
                else:
                    print(f"  ⚠ Saved medoid file not found, using first mesh")
                    template_file = os.path.join(vtk_dir, os.listdir(vtk_dir)[0])
                    ref_image_id = extract_image_id(os.path.basename(template_file))
            else:
                print(f"  ⚠ Could not extract Image ID from medoid, using first mesh")
                template_file = os.path.join(vtk_dir, os.listdir(vtk_dir)[0])
                ref_image_id = extract_image_id(os.path.basename(template_file))
        else:
            print(f"  ⚠ Medoid not found in metadata, using first mesh")
            template_file = os.path.join(vtk_dir, os.listdir(vtk_dir)[0])
            ref_image_id = extract_image_id(os.path.basename(template_file))
    else:
        print(f"  ⚠ Medoid column '{medoid_column}' not in metadata, using first mesh")
        template_file = os.path.join(vtk_dir, os.listdir(vtk_dir)[0])
        ref_image_id = extract_image_id(os.path.basename(template_file))
    
    # Check if Deformetrica output already exists
    reconstruction_pattern = os.path.join(deformetrica_output, "DeterministicAtlas__Reconstruction__*.vtk")
    existing_reconstructions = glob(reconstruction_pattern)
    template_output = os.path.join(deformetrica_output, "DeterministicAtlas__EstimatedParameters__Template_hippo.vtk")
    
    if len(existing_reconstructions) > 0 and os.path.exists(template_output):
        print(f"\nStep 3: Deformetrica Output Already Exists")
        print(f"  ✓ Found {len(existing_reconstructions)} existing reconstructed meshes")
        print(f"  ✓ Found existing template: {os.path.basename(template_output)}")
        print(f"  Skipping Deformetrica (already done), proceeding to volume correction...")
        reconstructed_files = sorted(existing_reconstructions)
    else:
        print(f"\nStep 3: Running Deformetrica Atlas Estimation")
        model, subject_ids_list, template_output = run_deformetrica_atlas(vtk_dir, template_file, deformetrica_output, max_iterations)
        print(f"  ✓ Deformetrica completed")
        
        # Find reconstructed meshes
        reconstructed_files = sorted(glob(reconstruction_pattern))
    
    # Save template to final output directory for easy access
    if template_output and os.path.exists(template_output):
        import shutil
        final_template_vtk = os.path.join(final_output_vtk, f"template_{version_name}.vtk")
        final_template_ply = os.path.join(final_output_ply, f"template_{version_name}.ply")
        
        # Copy VTK template
        shutil.copy(template_output, final_template_vtk)
        
        # Convert VTK to PLY manually (trimesh can't load VTK)
        vertices = []
        faces = []
        with open(template_output, 'r') as f:
            reading_points = False
            reading_polygons = False
            n_points = 0
            
            for line in f:
                line = line.strip()
                
                if line.startswith('POINTS'):
                    n_points = int(line.split()[1])
                    reading_points = True
                    reading_polygons = False
                    continue
                elif line.startswith('POLYGONS'):
                    reading_points = False
                    reading_polygons = True
                    continue
                elif line.startswith('CELL') or line.startswith('POINT_DATA') or line.startswith('METADATA'):
                    reading_points = False
                    reading_polygons = False
                    continue
                
                if reading_points and len(vertices) < n_points:
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        except:
                            pass
                
                elif reading_polygons:
                    parts = line.split()
                    if len(parts) >= 4 and parts[0] == '3':
                        try:
                            faces.append([int(parts[1]), int(parts[2]), int(parts[3])])
                        except:
                            pass
        
        # Save as PLY using trimesh
        template_mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
        template_mesh.export(final_template_ply)
        
        print(f"  ✓ Template saved: {os.path.basename(final_template_vtk)}")
        print(f"  ✓ Template saved: {os.path.basename(final_template_ply)}")
    else:
        final_template_vtk = None
        final_template_ply = None
    
    # Check vertex count
    if len(reconstructed_files) > 0:
        first_recon = reconstructed_files[0]
        with open(first_recon, 'r') as f:
            for line in f:
                if line.startswith('POINTS'):
                    n_points = int(line.split()[1])
                    print(f"  Output mesh vertex count: {n_points:,}")
                    if n_points < 5000 or n_points > 6000:
                        print(f"  ⚠ Warning: Vertex count outside typical range (5000-6000)")
                    break
    
    if len(reconstructed_files) == 0:
        print(f"✗ Error: No reconstructed meshes found")
        return None
    
    print(f"\nStep 4: Rescaling Meshes to Original Volumes")
    print(f"  Found {len(reconstructed_files)} reconstructed meshes")
    
    success_count = 0
    
    for recon_file in tqdm(reconstructed_files, desc="  Rescaling"):
        basename = os.path.basename(recon_file)
        
        # Extract Image ID from Deformetrica output
        # Format: DeterministicAtlas__Reconstruction__hippo__subject_I100370.vtk
        image_id = extract_image_id(basename)
        
        if not image_id or image_id not in original_volumes:
            print(f"  Warning: Subject {image_id} not found in original volumes")
            continue
        
        # Get original volume (in scaled space)
        target_volume = original_volumes[image_id]
        
        # Get original filename to preserve it
        original_ply_name = original_filenames[image_id]
        
        # Output paths with original filename
        final_vtk_file = os.path.join(final_output_vtk, original_ply_name.replace('.ply', '.vtk'))
        final_ply_file = os.path.join(final_output_ply, original_ply_name)
        
        # Rescale to match original volume (outputs both VTK and PLY)
        scale_factor, final_vol = rescale_mesh_to_volume(recon_file, final_vtk_file, final_ply_file, target_volume)
        
        success_count += 1
    
    print(f"  ✓ Rescaled {success_count}/{len(reconstructed_files)} meshes")
    
    # Check final files
    vtk_files = sorted(glob(os.path.join(final_output_vtk, "*.vtk")))
    ply_files = sorted(glob(os.path.join(final_output_ply, "*.ply")))
    
    print(f"\nStep 5: Summary")
    print(f"  Successfully processed: {success_count} meshes")
    print(f"  Final VTK files: {len(vtk_files)}")
    print(f"  Final PLY files: {len(ply_files)}")
    
    if len(vtk_files) > 0:
        print(f"\n  Sample output files:")
        print(f"    VTK: {os.path.basename(vtk_files[0])}")
        print(f"    PLY: {os.path.basename(ply_files[0])}")
    
    # Save summary
    summary = {
        'group': group_name,
        'version': version_name,
        'iterations': max_iterations,
        'template_medoid': ref_image_id,
        'template_vtk': os.path.basename(final_template_vtk) if final_template_vtk else None,
        'template_ply': os.path.basename(final_template_ply) if final_template_ply else None,
        'total_input': len(ply_files),
        'successful': success_count,
        'final_vtk': len(vtk_files),
        'final_ply': len(ply_files),
        'volume_unscale_factor': volume_unscale,
    }
    
    summary_df = pd.DataFrame([summary])
    summary_file = os.path.join(output_base, f"{version_name}_summary.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"\n  ✓ Summary saved to: {summary_file}")
    
    # Save README
    readme_file = os.path.join(output_base, f"{version_name}_README.txt")
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(f"{group_name.upper()} {version_name.upper()} - CORRESPONDENCE WITH VOLUME PRESERVATION\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"PROCESSING SUMMARY:\n")
        f.write(f"  • Input meshes: {len(ply_files)}\n")
        f.write(f"  • Initial template (medoid): {ref_image_id}\n")
        f.write(f"  • Learned template VTK: template_{version_name}.vtk\n")
        f.write(f"  • Learned template PLY: template_{version_name}.ply\n")
        f.write(f"  • Deformetrica iterations: {max_iterations}\n")
        f.write(f"  • Successfully processed: {success_count}\n")
        f.write(f"  • Final VTK files: {len(vtk_files)}\n")
        f.write(f"  • Final PLY files: {len(ply_files)}\n\n")
        f.write(f"TEMPLATE MESH:\n")
        f.write(f"  • The template represents the learned average shape of all subjects\n")
        f.write(f"  • Started from medoid ({ref_image_id}) and refined during atlas estimation\n")
        f.write(f"  • All subject meshes have the same topology as this template\n\n")
        f.write(f"MESH PROPERTIES:\n")
        f.write(f"  • Dense point correspondence: All meshes have identical topology\n")
        f.write(f"  • Vertex i corresponds to vertex i across all subjects\n")
        f.write(f"  • Volumes match original rigidly-aligned meshes\n")
        f.write(f"  • Original filenames preserved\n\n")
        f.write(f"VOLUME SCALING:\n")
        f.write(f"  • Volume unscale factor: {volume_unscale:.6e}\n")
        f.write(f"  • To convert to mm³: volume_mm3 = mesh.volume * {volume_unscale:.6e}\n\n")
        f.write(f"USAGE:\n")
        f.write(f"  • Use {version_name}_final_ply/*.ply for SpiralNet or mesh neural networks\n")
        f.write(f"  • Use {version_name}_final_vtk/*.vtk for visualization or VTK-based tools\n")
        f.write(f"  • All meshes have same vertex count and connectivity\n")
        f.write(f"  • Volumes are preserved for accurate biomarker analysis\n\n")
        f.write(f"DIRECTORIES:\n")
        f.write(f"  • {version_name}_vtk/: Converted rigidly-aligned meshes (input to Deformetrica)\n")
        f.write(f"  • {version_name}_deformetrica/: Raw Deformetrica output (correspondence, volumes may differ)\n")
        f.write(f"  • {version_name}_final_vtk/: Final meshes in VTK format\n")
        f.write(f"  • {version_name}_final_ply/: Final meshes in PLY format\n\n")
        f.write(f"NEXT STEPS:\n")
        f.write(f"  1. Validate correspondence by checking vertex counts are identical\n")
        f.write(f"  2. Visualize a few meshes to verify quality\n")
        f.write(f"  3. Proceed with downstream analysis (SpiralNet, PCA, etc.)\n")
    
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
    parser.add_argument('--iterations', type=int, default=50,
                       help='Max iterations for Deformetrica (default: 50, recommended for large datasets)')
    parser.add_argument('--base_dir', type=str, 
                       default='/home/jakaria/ADNI/ADNI_1/adni_processed/',
                       help='Base directory containing processed data')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ADNI HIPPOCAMPUS DENSE CORRESPONDENCE WITH VOLUME PRESERVATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Group: {args.group}")
    print(f"  Version: {args.version}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Base directory: {args.base_dir}")
    
    # Validate iteration count
    if args.iterations < 20:
        print(f"\n⚠ WARNING: {args.iterations} iterations is very low!")
        print(f"  Recommended minimum: 30 iterations")
        print(f"  Recommended for quality: 50+ iterations")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Process the specified group and version
    result = process_group(args.group, args.version, args.base_dir, args.iterations)
    
    if result:
        print(f"\n{'='*80}")
        print("PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"\nGroup: {result['group']}")
        print(f"Version: {result['version']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Successful meshes: {result['successful']}/{result['total_input']}")
        print(f"\nFinal meshes ready for downstream analysis!")
        print(f"  VTK: {result['final_vtk']} files")
        print(f"  PLY: {result['final_ply']} files")
    else:
        print(f"\n{'='*80}")
        print("PROCESSING FAILED")
        print(f"{'='*80}")
        sys.exit(1)


if __name__ == "__main__":
    main()
