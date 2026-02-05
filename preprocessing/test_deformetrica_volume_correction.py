#!/usr/bin/env python3
"""
Test Script: Deformetrica Volume Correction Validation

Tests the volume preservation pipeline on 10 samples from each group:
1. Compares original volumes (from XML ground truth)
2. Compares Deformetrica output volumes (after correspondence, before correction)
3. Compares final volumes (after volume correction)
4. Reports errors at each stage

Usage:
    python test_deformetrica_volume_correction.py
"""

import os
import sys
import re
import numpy as np
import pandas as pd
import trimesh
import xml.etree.ElementTree as ET
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
    """Calculate mesh volume"""
    if mesh_file.endswith('.vtk'):
        # Read VTK file manually
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
        
        # Create trimesh from vertices and faces
        mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
    else:
        mesh = trimesh.load(mesh_file)
    
    return abs(mesh.volume)


def extract_image_id(filename):
    """Extract Image ID from filename"""
    m = re.search(r'I\d+', filename)
    return m.group(0) if m else None


def find_xml_by_id(image_id, xml_folder):
    """Find XML file by Image ID"""
    for xml_file in glob(os.path.join(xml_folder, "*.xml")):
        if image_id in os.path.basename(xml_file):
            return xml_file
    return None


def extract_volumes_from_xml(xml_file):
    """Extract hippocampus volumes from XML file"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    ns = {'': 'http://www.loni.usc.edu/resource/xml/hippo/v1'}
    measures = root.findall('.//derivedMeasure', ns) or root.findall('.//derivedMeasure')
    
    vols = {}
    for m in measures:
        struct = m.find('.//measuredStructure', ns) or m.find('.//measuredStructure')
        if struct is not None and 'Hippocampus' in struct.text:
            hemi = m.find('.//measuredHemisphere', ns) or m.find('.//measuredHemisphere')
            val = m.find('.//measureValue', ns) or m.find('.//measureValue')
            if hemi is not None and val is not None:
                vols[hemi.text.strip()] = float(val.text)
    return vols


def get_nifti_ground_truth_volume(nifti_file):
    """Calculate ground truth volume from NIfTI segmentation"""
    img = nib.load(nifti_file)
    data = img.get_fdata()
    voxel_dims = img.header.get_zooms()
    voxel_volume = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]  # mm³ per voxel
    voxel_count = np.sum(data > 0)
    return voxel_count * voxel_volume


def rescale_mesh_to_volume(input_mesh, output_mesh, target_volume):
    """Rescale mesh uniformly to match target volume"""
    # Read VTK file if needed
    if input_mesh.endswith('.vtk'):
        vertices = []
        faces = []
        reading_points = False
        reading_polygons = False
        n_points = 0
        
        with open(input_mesh, 'r') as f:
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
        
        mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
    else:
        mesh = trimesh.load(input_mesh)
    
    current_volume = abs(mesh.volume)
    scale_factor = (target_volume / current_volume) ** (1.0 / 3.0)
    mesh.vertices *= scale_factor
    
    # Export based on output format
    if output_mesh.endswith('.vtk'):
        # Write VTK manually
        with open(output_mesh, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("vtk output\n")
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")
            f.write(f"POINTS {len(mesh.vertices)} float\n")
            for v in mesh.vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
            f.write(f"\nPOLYGONS {len(mesh.faces)} {len(mesh.faces) * 4}\n")
            for face in mesh.faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    else:
        mesh.export(output_mesh)
    
    return scale_factor, abs(mesh.volume)


def run_deformetrica_test(vtk_dir, template_file, output_dir):
    """Run Deformetrica atlas estimation"""
    vtk_files = sorted(glob(os.path.join(vtk_dir, "*.vtk")))
    
    deformetrica = dfca.Deformetrica(output_dir=str(output_dir), verbosity="ERROR")
    
    dataset_filenames = []
    subject_ids = []
    for vtk_file in vtk_files:
        # Use just the basename without extension as subject ID
        subject_id = os.path.basename(vtk_file).replace('.vtk', '')
        dataset_filenames.append([{"hippo": str(vtk_file)}])
        subject_ids.append(subject_id)
    
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
    
    estimator_options = {
        "optimization_method_type": "GradientAscent",
        "max_line_search_iterations": 10,
        "gpu_mode": "auto",
        "max_iterations": 100,
        "initial_step_size": 0.5,
        "convergence_tolerance": 1e-5,
        "save_every_n_iters": 10,
    }
    
    model_options = {
        "deformation_kernel_type": "keops",
        "deformation_kernel_width": 0.05,
        "number_of_timepoints": 20,  # Reduced for testing
    }
    
    model = deformetrica.estimate_deterministic_atlas(
        template_specifications,
        {"dataset_filenames": dataset_filenames, "subject_ids": subject_ids},
        estimator_options=estimator_options,
        model_options=model_options,
    )
    
    return model, subject_ids


def test_group(group_name, version_name, base_dir, test_dir, n_samples=10):
    """Test volume correction on n_samples from one group"""
    
    print(f"\n{'='*80}")
    print(f"TEST: {group_name.upper()} - {version_name.upper()} ({n_samples} samples)")
    print(f"{'='*80}")
    
    # Paths
    rigid_reg_dir = os.path.join(base_dir, f"{group_name}_hippocampus_ply_rigid_reg")
    ply_input_dir = os.path.join(rigid_reg_dir, version_name)
    metadata_file = os.path.join(rigid_reg_dir, "metadata.csv")
    
    # XML metadata directory (for ground truth volumes) - same as notebook
    xml_metadata_path = "/home/jakaria/ADNI/ADNI_1/ADNI_1_Hippocampal_Mask_IDA_Metadata/ADNI"
    
    output_base = os.path.join(test_dir, f"{group_name}_{version_name}")
    vtk_dir = os.path.join(output_base, "vtk")
    deformetrica_output = os.path.join(output_base, "deformetrica")
    final_output_vtk = os.path.join(output_base, "final_vtk")
    final_output_ply = os.path.join(output_base, "final_ply")
    
    os.makedirs(vtk_dir, exist_ok=True)
    os.makedirs(deformetrica_output, exist_ok=True)
    os.makedirs(final_output_vtk, exist_ok=True)
    os.makedirs(final_output_ply, exist_ok=True)
    
    # Check inputs
    if not os.path.exists(ply_input_dir):
        print(f"✗ Error: {ply_input_dir} not found")
        return None
    
    # Load metadata
    metadata_df = pd.read_csv(metadata_file)
    volume_unscale = metadata_df['volume_unscale_factor'].values[0]
    
    print(f"\nMetadata Info:")
    print(f"  Volume unscale factor: {volume_unscale:.6e}")
    print(f"  This converts PLY volumes to mm³")
    print(f"  XML metadata path: {xml_metadata_path}")
    
    # Check if XML directory exists and has files
    if os.path.exists(xml_metadata_path):
        xml_files = glob(os.path.join(xml_metadata_path, "*.xml"))
        print(f"  Found {len(xml_files)} XML files")
        if len(xml_files) > 0:
            print(f"  Sample XML file: {os.path.basename(xml_files[0])}")
    else:
        print(f"  ⚠ XML metadata directory does not exist!")
    
    # Select random n_samples
    ply_files = sorted(glob(os.path.join(ply_input_dir, "*.ply")))
    if len(ply_files) > n_samples:
        np.random.seed(42)
        ply_files = list(np.random.choice(ply_files, n_samples, replace=False))
    
    print(f"\nStep 1: Loading Ground Truth Volumes from XML + Converting to VTK")
    
    ground_truth_volumes = {}  # Key: Image ID
    original_volumes_ply = {}  # Key: Image ID
    original_filenames = {}  # Key: Image ID, Value: original PLY filename
    
    for ply_file in tqdm(ply_files, desc="  Processing"):
        basename = os.path.basename(ply_file).replace('.ply', '')
        # Remove group suffix (e.g., _left, _right, _combined)
        basename_clean = basename.replace(f'_{group_name}', '')
        
        # Extract Image ID (e.g., I40828)
        image_id = extract_image_id(basename_clean)
        if not image_id:
            print(f"  ⚠ Warning: No Image ID found in {basename}")
            continue
        
        # Store original filename
        original_filenames[image_id] = os.path.basename(ply_file)
        
        # Find corresponding XML file
        xml_file = find_xml_by_id(image_id, xml_metadata_path)
        if not xml_file:
            print(f"  ⚠ Warning: No XML found for {image_id}")
            continue
        
        # Get ground truth volume from XML
        xml_vols = extract_volumes_from_xml(xml_file)
        
        # Get appropriate volume based on group
        if group_name == 'left':
            gt_vol = xml_vols.get('Left')
        elif group_name == 'right':
            gt_vol = xml_vols.get('Right')
        elif group_name == 'combined':
            left_vol = xml_vols.get('Left')
            right_vol = xml_vols.get('Right')
            gt_vol = (left_vol + right_vol) if (left_vol and right_vol) else None
        else:
            gt_vol = None
        
        if gt_vol is None:
            print(f"  ⚠ Warning: No volume found in XML for {group_name}")
            continue
        
        # Store using Image ID as key (like notebook does)
        ground_truth_volumes[image_id] = gt_vol
        
        # Convert PLY to VTK (use Image ID in filename for consistency)
        vtk_file = os.path.join(vtk_dir, f"{image_id}.vtk")
        ply_to_vtk(ply_file, vtk_file)
        
        # Store PLY volume using Image ID as key
        ply_volume = calculate_mesh_volume(ply_file)
        original_volumes_ply[image_id] = ply_volume
    
    print(f"  ✓ Loaded {len(ground_truth_volumes)} ground truth volumes from NIfTI")
    print(f"  ✓ Converted {len(ply_files)} PLY files to VTK")
        # Check vertex count from first VTK file
    if len(glob(os.path.join(vtk_dir, "*.vtk"))) > 0:
        first_vtk = glob(os.path.join(vtk_dir, "*.vtk"))[0]
        with open(first_vtk, 'r') as f:
            for line in f:
                if line.startswith('POINTS'):
                    n_points = int(line.split()[1])
                    print(f"  Input mesh vertex count: {n_points}")
                    break
        # Use first mesh as template
    template_file = os.path.join(vtk_dir, os.listdir(vtk_dir)[0])
    
    print(f"\nStep 2: Running Deformetrica (reduced iterations for testing)")
    model, subject_ids_list = run_deformetrica_test(vtk_dir, template_file, deformetrica_output)
    print(f"  ✓ Deformetrica completed")
    
    # Check vertex count from first reconstruction
    recon_pattern = os.path.join(deformetrica_output, "DeterministicAtlas__Reconstruction__*.vtk")
    recon_files_check = glob(recon_pattern)
    if len(recon_files_check) > 0:
        first_recon = recon_files_check[0]
        with open(first_recon, 'r') as f:
            for line in f:
                if line.startswith('POINTS'):
                    n_points = int(line.split()[1])
                    print(f"  Output mesh vertex count: {n_points}")
                    if n_points < 5000 or n_points > 6000:
                        print(f"  ⚠ Warning: Vertex count {n_points} is outside 5000-6000 range")
                    else:
                        print(f"  ✓ Vertex count is within target range (5000-6000)")
                    break
    
    # Create mapping from Deformetrica output to original basenames
    subject_id_map = {}
    for sid in subject_ids_list:
        subject_id_map[sid] = sid  # They should be the same now
    
    # Find reconstructed meshes
    reconstruction_pattern = os.path.join(deformetrica_output, "DeterministicAtlas__Reconstruction__*.vtk")
    reconstructed_files = sorted(glob(reconstruction_pattern))
    
    print(f"\nStep 3: Volume Comparison (4 stages)")
    print(f"  Found {len(reconstructed_files)} reconstructed files")
    print(f"  Have {len(ground_truth_volumes)} ground truth volumes")
    
    # Debug: show first few keys
    if len(ground_truth_volumes) > 0:
        sample_keys = list(ground_truth_volumes.keys())[:2]
        print(f"  Sample ground truth keys: {sample_keys[:1] if sample_keys else 'none'}")
    
    results = []
    matched_count = 0
    
    for recon_file in tqdm(reconstructed_files, desc="  Processing"):
        basename = os.path.basename(recon_file)
        
        # Extract Image ID from Deformetrica output
        # Format: DeterministicAtlas__Reconstruction__hippo__subject_I100370.vtk
        # Extract just the Image ID (e.g., I100370)
        image_id = extract_image_id(basename)
        
        if not image_id:
            continue
        
        # Debug first file
        if matched_count == 0:
            print(f"\n  Example reconstruction file: {basename}")
            print(f"  Extracted image_id: {image_id}")
            if image_id in ground_truth_volumes:
                print(f"  ✓ Found match in ground truth!")
            else:
                print(f"  ✗ NOT found in ground truth")
                # Show what keys we have
                if len(ground_truth_volumes) > 0:
                    print(f"  Available keys (first): {list(ground_truth_volumes.keys())[0]}")
        
        if image_id not in ground_truth_volumes or image_id not in original_volumes_ply:
            continue
        
        matched_count += 1
        
        # Stage 1: Ground truth from XML (mm³)
        gt_vol_mm3 = ground_truth_volumes[image_id]
        
        # Stage 2: Rigidly-aligned PLY volume (mm³)
        ply_vol = original_volumes_ply[image_id]
        ply_vol_mm3 = ply_vol * volume_unscale
        
        # Stage 3: Deformetrica output volume (before correction)
        deformed_vol_ply = calculate_mesh_volume(recon_file)
        deformed_vol_mm3 = deformed_vol_ply * volume_unscale
        
        # Stage 4: After volume correction - save both VTK and PLY
        original_ply_name = original_filenames[image_id]
        
        final_vtk_file = os.path.join(final_output_vtk, original_ply_name.replace('.ply', '.vtk'))
        final_ply_file = os.path.join(final_output_ply, original_ply_name)
        
        scale_factor, final_vol_ply = rescale_mesh_to_volume(recon_file, final_vtk_file, ply_vol)
        
        # Convert VTK to PLY for final output - read VTK manually
        vertices = []
        faces = []
        with open(final_vtk_file, 'r') as f:
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
        
        mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))
        mesh.export(final_ply_file)
        
        final_vol_mm3 = final_vol_ply * volume_unscale
        
        # Calculate errors against ground truth
        ply_error = abs(ply_vol_mm3 - gt_vol_mm3) / gt_vol_mm3 * 100
        deformed_error = abs(deformed_vol_mm3 - gt_vol_mm3) / gt_vol_mm3 * 100
        final_error = abs(final_vol_mm3 - gt_vol_mm3) / gt_vol_mm3 * 100
        
        results.append({
            'image_id': image_id,
            'gt_vol_xml_mm3': gt_vol_mm3,
            'ply_vol_mm3': ply_vol_mm3,
            'deformed_vol_mm3': deformed_vol_mm3,
            'final_vol_mm3': final_vol_mm3,
            'ply_vs_gt_error_pct': ply_error,
            'deformed_vs_gt_error_pct': deformed_error,
            'final_vs_gt_error_pct': final_error,
            'scale_factor': scale_factor,
        })
    
    results_df = pd.DataFrame(results)
    
    # Check if we have results
    if len(results) == 0:
        print(f"\n✗ ERROR: No matching results found!")
        print(f"  This likely means subject IDs from Deformetrica don't match PLY basenames")
        return None
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"VOLUME COMPARISON RESULTS: {group_name.upper()} - {version_name.upper()}")
    print(f"{'='*80}")
    
    print(f"\nStage 1: Ground Truth (XML metadata):")
    print(f"  Mean: {results_df['gt_vol_xml_mm3'].mean():.2f} mm³")
    print(f"  Std:  {results_df['gt_vol_xml_mm3'].std():.2f} mm³")
    print(f"  Range: [{results_df['gt_vol_xml_mm3'].min():.2f}, {results_df['gt_vol_xml_mm3'].max():.2f}] mm³")
    
    print(f"\nStage 2: Rigidly-Aligned PLY (after batch processing):")
    print(f"  Mean volume: {results_df['ply_vol_mm3'].mean():.2f} mm³")
    print(f"  Error vs ground truth: {results_df['ply_vs_gt_error_pct'].mean():.2f}% ± {results_df['ply_vs_gt_error_pct'].std():.2f}%")
    if results_df['ply_vs_gt_error_pct'].mean() < 5.0:
        print(f"  Status: ✓ PLY volumes match ground truth")
    else:
        print(f"  Status: ⚠ PLY volumes differ from ground truth")
    
    print(f"\nStage 3: Deformetrica Output (before correction):")
    print(f"  Mean volume: {results_df['deformed_vol_mm3'].mean():.2f} mm³")
    print(f"  Error vs ground truth: {results_df['deformed_vs_gt_error_pct'].mean():.2f}% ± {results_df['deformed_vs_gt_error_pct'].std():.2f}%")
    print(f"  Status: ❌ Volumes CHANGED by Deformetrica")
    
    print(f"\nStage 4: After Volume Correction:")
    print(f"  Mean volume: {results_df['final_vol_mm3'].mean():.2f} mm³")
    print(f"  Error vs ground truth: {results_df['final_vs_gt_error_pct'].mean():.4f}% ± {results_df['final_vs_gt_error_pct'].std():.4f}%")
    print(f"  Max error: {results_df['final_vs_gt_error_pct'].max():.4f}%")
    
    if results_df['final_vs_gt_error_pct'].mean() < 0.1:
        print(f"  Status: ✓ EXCELLENT - Volumes match ground truth (< 0.1% error)")
    elif results_df['final_vs_gt_error_pct'].mean() < 1.0:
        print(f"  Status: ✓ PASS - Volumes match ground truth (< 1% error)")
    elif results_df['final_vs_gt_error_pct'].mean() < 5.0:
        print(f"  Status: ✓ ACCEPTABLE - Volumes close to ground truth (< 5% error)")
    else:
        print(f"  Status: ⚠ WARNING - High error (> 5%)")
    
    # Save detailed results
    results_file = os.path.join(output_base, "volume_comparison.csv")
    results_df.to_csv(results_file, index=False)
    print(f"\n  ✓ Detailed results saved to: {results_file}")
    print(f"  ✓ Final VTK files saved to: {final_output_vtk}")
    print(f"  ✓ Final PLY files saved to: {final_output_ply}")
    
    # Show sample filenames
    vtk_files = sorted(glob(os.path.join(final_output_vtk, "*.vtk")))
    ply_files = sorted(glob(os.path.join(final_output_ply, "*.ply")))
    if len(vtk_files) > 0:
        print(f"\n  Sample output files:")
        print(f"    VTK: {os.path.basename(vtk_files[0])}")
        print(f"    PLY: {os.path.basename(ply_files[0])}")
    
    # Print sample details
    print(f"\nSample Details (first 5):")
    print(results_df[['image_id', 'gt_vol_xml_mm3', 'ply_vol_mm3', 'deformed_vol_mm3', 'final_vol_mm3', 
                      'final_vs_gt_error_pct']].head().to_string(index=False))
    
    return {
        'group': group_name,
        'version': version_name,
        'n_samples': len(results),
        'gt_mean': results_df['gt_vol_xml_mm3'].mean(),
        'ply_error_mean': results_df['ply_vs_gt_error_pct'].mean(),
        'deformed_error_mean': results_df['deformed_vs_gt_error_pct'].mean(),
        'final_error_mean': results_df['final_vs_gt_error_pct'].mean(),
        'final_error_max': results_df['final_vs_gt_error_pct'].max(),
    }


def main():
    base_dir = "/home/jakaria/ADNI/ADNI_1/adni_processed/"
    test_dir = os.path.join(base_dir, "test")
    
    # Create test directory
    os.makedirs(test_dir, exist_ok=True)
    print(f"All test outputs will be stored in: {test_dir}\n")
    
    print("="*80)
    print("TEST: DEFORMETRICA VOLUME CORRECTION VALIDATION")
    print("Testing with 5 samples per group/version (100 iterations)")
    print("="*80)
    
    # Test configurations
    test_configs = [
        ('left', 'minimal'),
        ('left', 'minimal_smooth'),
        ('right', 'minimal'),
        ('right', 'minimal_smooth'),
        ('combined', 'minimal'),
        ('combined', 'minimal_smooth'),
    ]
    
    all_results = []
    
    for group, version in test_configs:
        result = test_group(group, version, base_dir, test_dir, n_samples=5)
        if result:
            all_results.append(result)
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY - ALL TESTS")
    print(f"{'='*80}")
    
    summary_df = pd.DataFrame(all_results)
    print(f"\n{summary_df.to_string(index=False)}")
    
    print(f"\n{'='*80}")
    print("CONCLUSIONS:")
    print(f"{'='*80}")
    print(f"1. Deformetrica CHANGES volumes (mean error: {summary_df['deformed_error_mean'].mean():.2f}%)")
    print(f"2. Volume correction RESTORES volumes (mean error: {summary_df['final_error_mean'].mean():.4f}%)")
    print(f"3. Final volumes match original within {summary_df['final_error_mean'].max():.4f}% average error")
    
    if summary_df['final_error_mean'].max() < 1.0:
        print(f"\n✓ TEST PASSED: Volume correction is working correctly!")
        print(f"  Ready to run full pipeline with batch_deformetrica_correspondence.py")
    else:
        print(f"\n⚠ WARNING: Review results before running full pipeline")


if __name__ == "__main__":
    main()
