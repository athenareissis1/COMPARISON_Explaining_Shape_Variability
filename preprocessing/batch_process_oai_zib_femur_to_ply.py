#!/usr/bin/env python3
"""
OAI-ZIB femur mask -> PLY meshes (ADNI-style minimal pipeline)

This script processes ONLY the IDs listed in:
  /home/jakaria/OAI-ZIB/merged_mask_labels.csv

For each ID, it finds the corresponding .mhd mask in either:
  - /home/jakaria/OAI-ZIB/classification/segmentation_masks
  - /home/jakaria/OAI-ZIB/segmentation/segmentation_masks

Then it extracts the femur bone (label=1) and creates volume-corrected meshes
using the same minimal steps used in the ADNI hippocampus pipeline:

  binarize -> pad -> toMesh(0.5) -> center -> volume correction -> global scale

Why global scaling?
- A single global scale factor normalizes mesh coordinates for stable downstream
  optimization (registration/correspondence/PCA). Relative size differences are
  preserved, since all subjects share the same global scale.

Recovering original measurements:
- Distance (mm) = PLY_distance * (1 / global_scale_factor)
- Volume (mm³)  = PLY_volume    * (1 / global_scale_factor)³

Notes:
- We also attempt a basic trimesh repair pass if a mesh is not watertight.
- Output meshes are centered at the origin (translation removed), like ADNI.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from glob import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shapeworks as sw
import SimpleITK as sitk
import trimesh
from tqdm import tqdm


# ==============================================================================
# Configuration (edit if needed)
# ==============================================================================
MERGED_CSV = "/home/jakaria/OAI-ZIB/merged_mask_labels.csv"
CLASSIFICATION_MASK_DIR = "/home/jakaria/OAI-ZIB/classification/segmentation_masks"
SEGMENTATION_MASK_DIR = "/home/jakaria/OAI-ZIB/segmentation/segmentation_masks"
OUTPUT_BASE_DIR = "/home/jakaria/OAI-ZIB/mesh_minimal"

FEMUR_LABEL = 1
PAD_VOXELS = 5
ISOVALUE = 0.5
GLOBAL_SCALE_BUFFER = 1.2  # matches ADNI script


# ==============================================================================
# Helpers
# ==============================================================================
@contextmanager
def suppress_stdout(out: bool = True, err: bool = False):
    """Suppress stdout/stderr to reduce console clutter (ShapeWorks can be noisy)."""
    with ExitStack() as stack:
        with open(os.devnull, "w") as null:
            if out:
                stack.enter_context(redirect_stdout(null))
            if err:
                stack.enter_context(redirect_stderr(null))
            yield


def read_mhd(mhd_path: str) -> Tuple[sitk.Image, np.ndarray]:
    itk_img = sitk.ReadImage(mhd_path)
    arr = sitk.GetArrayFromImage(itk_img)  # (z, y, x)
    return itk_img, arr


def extract_label_mask(seg_array: np.ndarray, label: int) -> np.ndarray:
    return (seg_array == label).astype(np.uint8)


def compute_seg_volume_mm3(mask_array: np.ndarray, spacing_xyz: Tuple[float, float, float]) -> float:
    voxel_volume = float(spacing_xyz[0]) * float(spacing_xyz[1]) * float(spacing_xyz[2])
    voxel_count = int(mask_array.sum())
    return voxel_count * voxel_volume


def compute_max_dimension_mm(mask_array: np.ndarray, spacing_xyz: Tuple[float, float, float]) -> Optional[float]:
    coords = np.where(mask_array > 0)
    if len(coords[0]) == 0:
        return None

    # coords are (z, y, x), spacing is (x, y, z)
    z_min, z_max = int(coords[0].min()), int(coords[0].max())
    y_min, y_max = int(coords[1].min()), int(coords[1].max())
    x_min, x_max = int(coords[2].min()), int(coords[2].max())

    extent_x = (x_max - x_min) * float(spacing_xyz[0])
    extent_y = (y_max - y_min) * float(spacing_xyz[1])
    extent_z = (z_max - z_min) * float(spacing_xyz[2])
    return float(max(extent_x, extent_y, extent_z))


def write_binary_mhd(mask_array: np.ndarray, reference_itk: sitk.Image, out_path: str) -> str:
    itk_mask = sitk.GetImageFromArray(mask_array)  # expects (z, y, x)
    itk_mask.SetSpacing(reference_itk.GetSpacing())
    itk_mask.SetOrigin(reference_itk.GetOrigin())
    itk_mask.SetDirection(reference_itk.GetDirection())
    sitk.WriteImage(itk_mask, out_path)
    return out_path


def load_trimesh_mesh(mesh_path: str) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    return mesh


def repair_trimesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Best-effort cleanup to improve watertightness/volume computations."""
    # Basic cleanup
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()
    mesh.merge_vertices()

    # Try to close small holes and fix orientation
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_normals(mesh)
    return mesh


def safe_mesh_volume(mesh: trimesh.Trimesh) -> float:
    """Return a non-negative volume estimate."""
    vol = float(mesh.volume)
    return abs(vol)


def patient_id_from_mhd(mhd_path: str) -> str:
    # Example: 9008561.segmentation_masks.mhd -> 9008561
    return os.path.basename(mhd_path).split(".")[0]


def build_mask_index(mask_dir: str) -> Dict[str, str]:
    """Build {patient_id: mhd_path} mapping for a mask directory."""
    index: Dict[str, str] = {}
    for mhd_path in glob(os.path.join(mask_dir, "*.mhd")):
        pid = patient_id_from_mhd(mhd_path)
        index[pid] = mhd_path
    return index


def resolve_mask_path(pid: str, cls_index: Dict[str, str], seg_index: Dict[str, str]) -> Optional[str]:
    """Prefer classification masks; fallback to segmentation masks."""
    if pid in cls_index:
        return cls_index[pid]
    if pid in seg_index:
        return seg_index[pid]
    return None


# ==============================================================================
# Processing
# ==============================================================================
def compute_global_scale(mhd_files: List[str]) -> Dict[str, float]:
    print(f"\n{'='*80}")
    print("STEP 1: Computing global scale factor (ADNI-style)")
    print(f"{'='*80}")

    max_dimensions: List[float] = []
    for mhd_path in tqdm(mhd_files, desc="Finding max femur dimension"):
        try:
            itk_img, arr = read_mhd(mhd_path)
            femur_mask = extract_label_mask(arr, FEMUR_LABEL)
            max_dim = compute_max_dimension_mm(femur_mask, itk_img.GetSpacing())
            if max_dim is not None:
                max_dimensions.append(max_dim)
        except Exception as e:
            print(f"  Warning: failed dimension read for {os.path.basename(mhd_path)}: {e}")

    if not max_dimensions:
        raise RuntimeError("Could not compute max dimensions for any femur masks.")

    global_max_dimension = float(max(max_dimensions))
    min_dimension = float(min(max_dimensions))
    mean_dimension = float(np.mean(max_dimensions))
    std_dimension = float(np.std(max_dimensions))

    global_max_dimension_buffered = global_max_dimension * GLOBAL_SCALE_BUFFER
    global_scale = 1.0 / global_max_dimension_buffered
    volume_unscale = (1.0 / global_scale) ** 3

    print("\n✓ Global scale calculation complete:")
    print(f"  Dimension range: {min_dimension:.2f} - {global_max_dimension:.2f} mm")
    print(f"  Mean dimension: {mean_dimension:.2f} ± {std_dimension:.2f} mm")
    print(f"  Max dimension (buffered): {global_max_dimension_buffered:.2f} mm")
    print(f"  Global scale factor: {global_scale:.8f}")
    print(f"  Volume unscale factor: {volume_unscale:.6e}")
    print("\nTo recover original measurements from PLY files:")
    print(f"  • Distance (mm) = PLY_distance × {1/global_scale:.6f}")
    print(f"  • Volume (mm³) = PLY_volume × {volume_unscale:.6e}")

    return {
        "dimension_min_mm": min_dimension,
        "dimension_max_mm": global_max_dimension,
        "dimension_mean_mm": mean_dimension,
        "dimension_std_mm": std_dimension,
        "dimension_max_buffered_mm": float(global_max_dimension_buffered),
        "global_scale_factor": float(global_scale),
        "distance_unscale_factor": float(1.0 / global_scale),
        "volume_unscale_factor": float(volume_unscale),
    }


def process_single_mask(
    mhd_path: str,
    global_scale: float,
    volume_unscale: float,
    binary_dir: str,
    mesh_dir: str,
) -> Optional[Dict]:
    pid = patient_id_from_mhd(mhd_path)

    itk_img, arr = read_mhd(mhd_path)
    spacing_xyz = itk_img.GetSpacing()

    femur_mask = extract_label_mask(arr, FEMUR_LABEL)
    if int(femur_mask.sum()) == 0:
        return None

    seg_volume_mm3 = compute_seg_volume_mm3(femur_mask, spacing_xyz)

    # Persist binary mask for reproducibility (ShapeWorks reads from disk)
    femur_mhd = os.path.join(binary_dir, f"{pid}_femur.mhd")
    write_binary_mhd(femur_mask, itk_img, femur_mhd)

    # ShapeWorks minimal pipeline (ADNI-style)
    with suppress_stdout():
        shape_seg = sw.Image(femur_mhd)
        shape_seg.binarize(0)
        shape_seg.pad(PAD_VOXELS, 0)
        mesh_sw = shape_seg.toMesh(ISOVALUE)
        center = mesh_sw.center()
        mesh_sw.translate(list(-center))

        tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
        tmp_path = tmp.name
        tmp.close()
        mesh_sw.write(tmp_path)

    mesh = load_trimesh_mesh(tmp_path)
    os.unlink(tmp_path)

    was_watertight = bool(mesh.is_watertight)
    if not was_watertight:
        mesh = repair_trimesh(mesh)

    is_watertight = bool(mesh.is_watertight)
    is_volume = bool(mesh.is_volume)

    mesh_volume_mm3 = safe_mesh_volume(mesh)
    if not np.isfinite(mesh_volume_mm3) or mesh_volume_mm3 <= 0:
        return None

    # Volume correction to match segmentation volume
    vol_correction = float((seg_volume_mm3 / mesh_volume_mm3) ** (1.0 / 3.0))
    mesh.apply_scale(vol_correction)

    # Global scaling (dataset normalization)
    mesh.apply_scale(global_scale)

    # Export final mesh
    out_mesh = os.path.join(mesh_dir, f"{pid}_femur.ply")
    mesh.export(out_mesh)

    # Volumes
    mesh_volume_scaled = safe_mesh_volume(mesh)
    mesh_volume_unscaled_mm3 = mesh_volume_scaled * volume_unscale

    return {
        "patient_id": pid,
        "input_mhd": mhd_path,
        "binary_femur_mhd": femur_mhd,
        "output_ply": out_mesh,
        "seg_volume_mm3": float(seg_volume_mm3),
        "mesh_volume_scaled": float(mesh_volume_scaled),
        "mesh_volume_unscaled_mm3": float(mesh_volume_unscaled_mm3),
        "volume_correction": float(vol_correction),
        "was_watertight": was_watertight,
        "is_watertight": is_watertight,
        "is_volume": is_volume,
    }


def main() -> None:
    print("=" * 80)
    print("OAI-ZIB FEMUR -> PLY (ADNI-STYLE MINIMAL + VOLCORR)")
    print("=" * 80)

    # Load merged labels (source of truth for which IDs to process)
    merged_df = pd.read_csv(MERGED_CSV)
    merged_df["patient_id"] = merged_df["patient_id"].astype(str)
    before = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=["patient_id"])
    if len(merged_df) != before:
        print(f"⚠ Dropped duplicate IDs in merged CSV: {before - len(merged_df)}")

    target_ids = merged_df["patient_id"].tolist()
    print(f"\nMerged CSV: {MERGED_CSV}")
    print(f"Target IDs: {len(target_ids)}")

    # Build mask indices for both folders
    cls_index = build_mask_index(CLASSIFICATION_MASK_DIR)
    seg_index = build_mask_index(SEGMENTATION_MASK_DIR)

    print(f"\nClassification masks: {len(cls_index)}")
    print(f"Segmentation masks:  {len(seg_index)}")

    # Resolve mask paths for target IDs
    id_to_mhd: Dict[str, str] = {}
    missing_ids: List[str] = []
    for pid in target_ids:
        mhd_path = resolve_mask_path(pid, cls_index, seg_index)
        if mhd_path is None:
            missing_ids.append(pid)
        else:
            id_to_mhd[pid] = mhd_path

    mhd_files = list(id_to_mhd.values())
    print(f"\nResolved masks for IDs: {len(mhd_files)}")
    print(f"Missing IDs (no mask in either folder): {len(missing_ids)}")
    if missing_ids:
        print("Missing IDs:")
        print(missing_ids)

    if not mhd_files:
        raise SystemExit("No masks resolved from merged CSV. Check inputs.")

    # Output structure
    minimal_dir = OUTPUT_BASE_DIR  # all meshes
    healthy_dir = os.path.join(OUTPUT_BASE_DIR, "healthy")
    diseased_dir = os.path.join(OUTPUT_BASE_DIR, "diseased")
    binary_dir = os.path.join(OUTPUT_BASE_DIR, "binary_masks_femur")
    os.makedirs(minimal_dir, exist_ok=True)
    os.makedirs(healthy_dir, exist_ok=True)
    os.makedirs(diseased_dir, exist_ok=True)
    os.makedirs(binary_dir, exist_ok=True)

    print(f"\nOutput directory: {OUTPUT_BASE_DIR}")
    print(f"  Meshes:  {minimal_dir}")
    print(f"  Binaries:{binary_dir}")

    # Step 1: global scale
    scale_meta = compute_global_scale(mhd_files)
    global_scale = scale_meta["global_scale_factor"]
    volume_unscale = scale_meta["volume_unscale_factor"]

    # Step 2: process all
    print(f"\n{'='*80}")
    print("STEP 2: Processing all femur masks")
    print(f"{'='*80}")

    records: List[Dict] = []
    failed: List[str] = []

    for pid, mhd_path in tqdm(id_to_mhd.items(), desc="Meshing femurs"):
        try:
            row = merged_df.loc[merged_df["patient_id"] == pid]
            disease_label = None
            if not row.empty:
                disease_label = int(row["disease_diagnosis"].iloc[0])

            rec = process_single_mask(
                mhd_path=mhd_path,
                global_scale=global_scale,
                volume_unscale=volume_unscale,
                binary_dir=binary_dir,
                mesh_dir=minimal_dir,
            )
            if rec is None:
                failed.append(mhd_path)
            else:
                # Attach label fields from merged CSV (if present)
                if not row.empty:
                    rec.update(
                        {
                            "disease_diagnosis": int(row["disease_diagnosis"].iloc[0]),
                            "kl_grade": int(row["kl_grade"].iloc[0]),
                            "gender": int(row["gender"].iloc[0]),
                            "age": int(row["age"].iloc[0]),
                        }
                    )

                # Copy mesh into healthy/diseased folders
                if disease_label is not None:
                    if disease_label == 1:
                        shutil.copy2(rec["output_ply"], os.path.join(diseased_dir, os.path.basename(rec["output_ply"])))
                    else:
                        shutil.copy2(rec["output_ply"], os.path.join(healthy_dir, os.path.basename(rec["output_ply"])))

                records.append(rec)
        except Exception as e:
            print(f"\n  Error processing {os.path.basename(mhd_path)}: {e}")
            failed.append(mhd_path)

    df = pd.DataFrame(records)
    total = len(mhd_files)
    success = len(records)
    n_failed = len(failed)

    print("\n✓ Processing complete:")
    print(f"  Success: {success}/{total}")
    print(f"  Failed:  {n_failed}")

    if success > 0:
        print(f"  Watertight before repair: {int(df['was_watertight'].sum())}/{success}")
        print(f"  Watertight after repair:  {int(df['is_watertight'].sum())}/{success}")
        print(f"  Closed volume meshes:     {int(df['is_volume'].sum())}/{success}")
        print(f"  Volume correction factors: {df['volume_correction'].mean():.4f} ± {df['volume_correction'].std():.4f}")

    # Step 3: save metadata + per-mesh CSV
    print(f"\n{'='*80}")
    print("STEP 3: Saving metadata")
    print(f"{'='*80}")

    metrics_csv = os.path.join(OUTPUT_BASE_DIR, "mesh_metrics.csv")
    df.to_csv(metrics_csv, index=False)
    print(f"✓ Per-mesh metrics saved to: {metrics_csv}")

    metadata = {
        **scale_meta,
        "merged_csv": MERGED_CSV,
        "classification_mask_dir": CLASSIFICATION_MASK_DIR,
        "segmentation_mask_dir": SEGMENTATION_MASK_DIR,
        "output_dir": OUTPUT_BASE_DIR,
        "label_femur": FEMUR_LABEL,
        "pad_voxels": PAD_VOXELS,
        "isovalue": ISOVALUE,
        "total_files": total,
        "success": success,
        "failed": n_failed,
        "watertight_before_repair": int(df["was_watertight"].sum()) if success > 0 else 0,
        "watertight_after_repair": int(df["is_watertight"].sum()) if success > 0 else 0,
        "closed_volume_meshes": int(df["is_volume"].sum()) if success > 0 else 0,
        "vol_corr_mean": float(df["volume_correction"].mean()) if success > 0 else None,
        "vol_corr_std": float(df["volume_correction"].std()) if success > 0 else None,
    }
    metadata_df = pd.DataFrame({k: [v] for k, v in metadata.items()})
    metadata_csv = os.path.join(OUTPUT_BASE_DIR, "metadata.csv")
    metadata_df.to_csv(metadata_csv, index=False)
    print(f"✓ Metadata saved to: {metadata_csv}")

    readme_path = os.path.join(OUTPUT_BASE_DIR, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("OAI-ZIB FEMUR PLY FILES - METADATA\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Merged CSV:       {MERGED_CSV}\n")
        f.write(f"Classification masks: {CLASSIFICATION_MASK_DIR}\n")
        f.write(f"Segmentation masks:  {SEGMENTATION_MASK_DIR}\n")
        f.write(f"Output directory: {OUTPUT_BASE_DIR}\n\n")
        f.write(f"Total masks: {total}\n")
        f.write(f"Success:     {success}\n")
        f.write(f"Failed:      {n_failed}\n\n")
        f.write("GLOBAL SCALE INFORMATION\n")
        f.write("=" * 80 + "\n")
        f.write(f"Global scale factor: {global_scale:.8f}\n")
        f.write(f"Volume unscale factor: {volume_unscale:.6e}\n\n")
        f.write("RECOVERING ORIGINAL MEASUREMENTS\n")
        f.write("=" * 80 + "\n")
        f.write("Distance (mm):\n")
        f.write(f"  original_distance_mm = ply_distance × {1/global_scale:.6f}\n\n")
        f.write("Volume (mm³):\n")
        f.write(f"  original_volume_mm3 = ply_volume × {volume_unscale:.6e}\n\n")
        f.write("VOLUME CORRECTION\n")
        f.write("=" * 80 + "\n")
        f.write("Meshes are volume-corrected to match segmentation volumes computed from voxels.\n")
        if success > 0:
            f.write(
                f"Volume correction factor mean ± std: "
                f"{df['volume_correction'].mean():.4f} ± {df['volume_correction'].std():.4f}\n\n"
            )
        f.write("FILES\n")
        f.write("=" * 80 + "\n")
        f.write("*.ply                - Final PLY meshes (minimal + volcorr + global scale)\n")
        f.write("healthy/             - Copies of healthy meshes (disease_diagnosis=0)\n")
        f.write("diseased/            - Copies of diseased meshes (disease_diagnosis=1)\n")
        f.write("binary_masks_femur/  - Saved binary femur masks used as mesh input\n")
        f.write("mesh_metrics.csv     - Per-mesh volumes, correction factors, watertight flags\n")
        f.write("metadata.csv         - Global scale factors + processing summary\n")
        f.write("README.txt           - This file\n")

    print(f"✓ README saved to: {readme_path}")

    if failed:
        failed_txt = os.path.join(OUTPUT_BASE_DIR, "failed_cases.txt")
        with open(failed_txt, "w", encoding="utf-8") as f:
            for p in failed:
                f.write(p + "\n")
        print(f"⚠ Failed cases saved to: {failed_txt}")


if __name__ == "__main__":
    main()
