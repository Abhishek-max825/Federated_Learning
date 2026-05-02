"""
Split PTBXL ECG Dataset into 3 Hospital Clients for Federated Learning

This script:
1. Reads PTBXL metadata
2. Filters to Normal/Arrhythmia binary classification
3. Splits records deterministically into 3 hospital folders
4. Copies WFDB files (.dat + .hea) to each hospital
5. Creates hospital-specific metadata CSVs

Usage:
    python split_hospitals.py

Output:
    ecg_dataset/hospital_1/  - First 1/3 of records
    ecg_dataset/hospital_2/  - Middle 1/3 of records
    ecg_dataset/hospital_3/  - Last 1/3 of records
"""

import pandas as pd
import numpy as np
import shutil
import os
import ast
from pathlib import Path
from collections import defaultdict

# Configuration
SCRIPT_DIR = Path(__file__).parent
PTBXL_CSV = SCRIPT_DIR / "ptbxl_database.csv"
ECG_RECORDS_PATH = SCRIPT_DIR / "records100"

# Verify paths exist
if not PTBXL_CSV.exists():
    raise FileNotFoundError(f"PTBXL CSV not found: {PTBXL_CSV}")
if not ECG_RECORDS_PATH.exists():
    raise FileNotFoundError(f"Records folder not found: {ECG_RECORDS_PATH}")

# Hospital output directories
HOSPITALS = {
    1: SCRIPT_DIR / "hospital_1",
    2: SCRIPT_DIR / "hospital_2",
    3: SCRIPT_DIR / "hospital_3",
}

# SCP codes mapping
# NORM = clearly normal rhythm
NORMAL_CODES = {"NORM", "SR"}

# All abnormal/arrhythmia codes from PTB-XL SCP standard
# We treat everything that is NOT normal as arrhythmia for binary classification
ARRHYTHMIA_CODES = {
    # Rhythm abnormalities
    "AFIB", "AFLT", "SVES", "VES", "PVC", "PAC", "APC",
    "SVTAC", "VTAC", "VFIB", "VFLU", "SBRAD", "STACH", "SARRH", "SVARR",
    "TACHY", "BRADY", "BIGU", "TRIGU",
    # Conduction abnormalities
    "LBBB", "RBBB", "ILBBB", "IRBBB", "LPFB", "LAFB", "WPW",
    "1AVB", "2AVB", "3AVB", "IVCD",
    # ST/T abnormalities
    "NST_", "NDT", "DIG", "LNGQT", "ABQRS",
    # Myocardial infarction
    "IMI", "ASMI", "ILMI", "AMI", "ALMI", "INJAS", "LMI", "RMI",
    # Hypertrophy
    "LVH", "RVH", "SEHYP",
    # Other
    "INJAL", "ISCAL", "ISCAN", "ISCIN", "ISCLA", "ISC_",
    "INJIN", "INJLA", "PMI"
}


def parse_scp_codes(scp_string):
    """Parse SCP codes from string representation."""
    try:
        scp_dict = ast.literal_eval(scp_string)
        if not scp_dict:
            return None
        primary_code = max(scp_dict, key=scp_dict.get)
        return primary_code
    except:
        return None


def classify_ecg_binary(scp_code):
    """
    Classify ECG as binary: 0=Normal, 1=Arrhythmia
    
    Args:
        scp_code: Primary SCP code
        
    Returns:
        0 for Normal, 1 for Arrhythmia, None for unknown
    """
    if scp_code is None:
        return None
    
    if scp_code in NORMAL_CODES:
        return 0
    elif scp_code in ARRHYTHMIA_CODES:
        return 1
    else:
        # Any other code not explicitly in normal set → treat as arrhythmia
        # This ensures all non-normal ECGs are captured
        return 1


def assign_hospital_by_record_id(ecg_id):
    """
    Assign hospital by round-robin on ecg_id so each hospital
    gets an equal share regardless of which folders exist on disk.
    """
    try:
        record_num = int(ecg_id)
    except:
        record_num = hash(str(ecg_id)) % 21837
    
    return (record_num % 3) + 1  # 1, 2, or 3


def copy_wfdb_files(record_id, filename_lr, hospital_dir):
    """
    Copy WFDB files (.dat and .hea) for a record.
    
    Args:
        record_id: ECG ID (e.g., '00001')
        filename_lr: Relative path from ptbxl (e.g., '00000/00001_lr')
        hospital_dir: Destination hospital directory
        
    Returns:
        bool: True if files copied successfully
    """
    # Source paths
    # filename_lr already includes 'records100/' prefix, so use SCRIPT_DIR as base
    src_base = SCRIPT_DIR / filename_lr
    src_dat = str(src_base) + ".dat"
    src_hea = str(src_base) + ".hea"
    
    # Destination: hospital_dir/records100/00000/00001_lr.dat
    # filename_lr is like 'records100/00000/00001_lr', so use it directly
    dst_path = hospital_dir / filename_lr
    dst_folder = dst_path.parent
    dst_folder.mkdir(parents=True, exist_ok=True)
    
    dst_dat = dst_folder / (dst_path.name + ".dat")
    dst_hea = dst_folder / (dst_path.name + ".hea")
    
    success = True
    
    try:
        if os.path.exists(src_dat):
            shutil.copy2(src_dat, dst_dat)
        else:
            print(f"  Warning: Missing {src_dat}")
            success = False
            
        if os.path.exists(src_hea):
            shutil.copy2(src_hea, dst_hea)
        else:
            print(f"  Warning: Missing {src_hea}")
            success = False
            
    except Exception as e:
        print(f"  Error copying {record_id}: {e}")
        success = False
    
    return success


def split_dataset():
    """Main function to split dataset into hospitals."""
    
    print("=" * 60)
    print("ECG Dataset Split for Federated Learning")
    print("=" * 60)
    
    # Load metadata
    print("\n[1/5] Loading PTBXL metadata...")
    df = pd.read_csv(PTBXL_CSV)
    print(f"  Total records in database: {len(df)}")
    
    # Parse SCP codes and classify
    print("\n[2/5] Parsing SCP codes and classifying...")
    df["primary_scp"] = df["scp_codes"].apply(parse_scp_codes)
    df["label"] = df["primary_scp"].apply(classify_ecg_binary)
    
    # Filter to binary classifiable records
    df_classified = df.dropna(subset=["label"]).copy()
    df_classified["label"] = df_classified["label"].astype(int)
    
    print(f"  Classifiable records: {len(df_classified)}")
    print(f"    Normal (0): {sum(df_classified['label'] == 0)}")
    print(f"    Arrhythmia (1): {sum(df_classified['label'] == 1)}")
    
    # Filter to only records with actual files on disk
    print("\n[2.5/5] Filtering to records with existing .dat/.hea files...")
    def files_exist(filename_lr):
        base = SCRIPT_DIR / filename_lr
        return os.path.exists(str(base) + ".dat") and os.path.exists(str(base) + ".hea")
    
    df_classified = df_classified[df_classified["filename_lr"].apply(files_exist)].copy()
    print(f"  Records with files on disk: {len(df_classified)}")
    print(f"    Normal (0): {sum(df_classified['label'] == 0)}")
    print(f"    Arrhythmia (1): {sum(df_classified['label'] == 1)}")

    # Assign hospitals
    print("\n[3/5] Assigning records to hospitals with balanced sampling...")
    df_classified["hospital_id"] = df_classified["ecg_id"].apply(assign_hospital_by_record_id)
    
    # Balance each hospital: undersample Normal to match Arrhythmia count
    balanced_dfs = []
    for hosp_id in [1, 2, 3]:
        hosp_df = df_classified[df_classified["hospital_id"] == hosp_id]
        n_normal = sum(hosp_df['label'] == 0)
        n_arrhy = sum(hosp_df['label'] == 1)
        print(f"  Hospital {hosp_id} before balance: Normal={n_normal}, Arrhythmia={n_arrhy}")
        
        normal_df = hosp_df[hosp_df['label'] == 0]
        arrhy_df = hosp_df[hosp_df['label'] == 1]
        
        # Undersample majority class to match minority
        target = min(n_normal, n_arrhy)
        if len(arrhy_df) > target:
            arrhy_df = arrhy_df.sample(n=target, random_state=42)
        
        hosp_balanced = pd.concat([normal_df, arrhy_df]).sample(frac=1, random_state=42)
        balanced_dfs.append((hosp_id, hosp_balanced))
        print(f"  Hospital {hosp_id} after balance:  Normal={sum(hosp_balanced['label']==0)}, Arrhythmia={sum(hosp_balanced['label']==1)}")
    
    # Create hospital directories and copy files
    print("\n[4/5] Writing hospital metadata (no file copying — shared records100 folder)...")
    
    for hosp_id, hosp_df in balanced_dfs:
        hosp_dir = HOSPITALS[hosp_id]
        hosp_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n  Hospital {hosp_id}: {len(hosp_df)} records (Normal={sum(hosp_df['label']==0)}, Arrhythmia={sum(hosp_df['label']==1)})")
        
        # Save hospital metadata
        metadata_cols = [
            "ecg_id", "patient_id", "age", "sex", "height", "weight",
            "recording_date", "report", "primary_scp", "label",
            "filename_lr", "filename_hr", "heart_axis",
            "infarction_stadium1", "infarction_stadium2",
            "baseline_drift", "static_noise", "burst_noise",
            "electrodes_problems", "extra_beats", "pacemaker"
        ]
        available_cols = [c for c in metadata_cols if c in hosp_df.columns]
        hosp_metadata = hosp_df[available_cols].copy()
        
        metadata_path = hosp_dir / "metadata.csv"
        hosp_metadata.to_csv(metadata_path, index=False)
        print(f"    Metadata saved: {metadata_path}")
    
    # Create summary report
    print("\n[5/5] Creating summary report...")
    summary_path = SCRIPT_DIR / "split_summary.txt"
    with open(summary_path, "w") as f:
        f.write("ECG Dataset Split Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Source: {PTBXL_CSV}\n")
        f.write(f"Total classifiable records: {len(df_classified)}\n\n")
        
        for hosp_id, hosp_df in balanced_dfs:
            f.write(f"Hospital {hosp_id}:\n")
            f.write(f"  Records: {len(hosp_df)}\n")
            f.write(f"  Normal: {sum(hosp_df['label'] == 0)}\n")
            f.write(f"  Arrhythmia: {sum(hosp_df['label'] == 1)}\n")
            f.write(f"  Location: {HOSPITALS[hosp_id]}\n\n")
    
    print(f"  Summary saved: {summary_path}")
    
    print("\n" + "=" * 60)
    print("Split Complete!")
    print("=" * 60)
    print(f"\nHospital directories created:")
    for hosp_id in [1, 2, 3]:
        print(f"  {HOSPITALS[hosp_id]}")
    
    print("\nEach hospital can now train independently using:")
    print("  1. metadata.csv for labels")
    print("  2. records100/ for WFDB signal files")
    print("\nNo data sharing between hospitals - only model weights!")


if __name__ == "__main__":
    split_dataset()
