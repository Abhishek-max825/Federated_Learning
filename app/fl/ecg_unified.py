"""
Unified ECG Pipeline Module

This module provides a SINGLE, CONSISTENT pipeline for:
- Training (hospital local training)
- Inference (doctor predictions)
- Federated Learning (data loading)

CRITICAL: This is the ONLY ECG processing module. No separate paths for training vs inference.

Pipeline Flow:
    WFDB File → Load → Preprocess → (12-lead signal, Lead II) → Model/Clinical Analysis

Usage:
    from app.fl.ecg_unified import ECGUnifiedPipeline
    
    pipeline = ECGUnifiedPipeline()
    
    # For training/inference
    signal_12lead, lead_ii = pipeline.load_and_preprocess(filepath)
    
    # For model input (add batch dimension)
    model_input = pipeline.prepare_for_model(signal_12lead)  # (1, 12, 1000)
"""

import os
import numpy as np
import pandas as pd
import wfdb
from scipy import signal
from typing import Tuple, Optional, Dict, List
from pathlib import Path


class ECGUnifiedPipeline:
    """
    Unified ECG processing pipeline.
    
    SAME preprocessing for training and inference.
    Supports WFDB format (.dat + .hea files).
    """
    
    # Standard 12-lead ECG configuration (PTB-XL standard)
    LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    LEAD_II_INDEX = 1  # Lead II is at index 1 in PTB-XL
    
    # Sampling parameters
    FS = 500  # Hz (PTB-XL 500Hz resampled)
    TARGET_LENGTH = 1000  # samples (2 seconds at 500Hz)
    
    # Preprocessing parameters
    HIGHPASS_CUTOFF = 0.5  # Hz (remove baseline wander)
    
    def __init__(self, fs: int = 500, target_length: int = 1000):
        """
        Initialize unified pipeline.
        
        Args:
            fs: Sampling frequency in Hz (default 500)
            target_length: Target signal length in samples (default 1000)
        """
        self.fs = fs
        self.target_length = target_length
        self._design_filter()
    
    def _design_filter(self):
        """Design high-pass Butterworth filter for baseline removal."""
        nyquist = self.fs / 2
        normal_cutoff = self.HIGHPASS_CUTOFF / nyquist
        self.b_high, self.a_high = signal.butter(
            N=4,  # 4th order
            Wn=normal_cutoff,
            btype='high',
            analog=False
        )
    
    def load_wfdb(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Load ECG signal from WFDB format.
        
        Args:
            filepath: Path to WFDB file (without extension, e.g., '00001_lr')
            
        Returns:
            Tuple of (signal_data, fs) where signal_data is (timesteps x leads) and fs is the actual sampling frequency.
        """
        try:
            # wfdb.rdsamp reads both .dat and .hea
            record = wfdb.rdsamp(filepath)
            signal_data = record[0]  # (timesteps, leads)
            fields = record[1]
            fs = int(fields.get('fs', 100))  # actual WFDB sampling frequency
            
            # Ensure float32
            signal_data = signal_data.astype(np.float32)
            
            return signal_data, fs
            
        except Exception as e:
            raise ValueError(f"Failed to load WFDB file {filepath}: {str(e)}")
    
    def resample_to_target(self, signal_data: np.ndarray, original_fs: int = 100) -> np.ndarray:
        """
        Resample signal to target length (500Hz, 1000 samples).
        
        PTB-XL low-res is 100Hz, needs upsampling to 500Hz.
        
        Args:
            signal_data: Raw signal (timesteps, leads)
            original_fs: Original sampling frequency (default 100 for PTB-XL low-res)
            
        Returns:
            Resampled signal (1000, leads)
        """
        num_leads = signal_data.shape[1]
        
        # Calculate target samples
        duration = len(signal_data) / original_fs
        target_samples = int(duration * self.fs)
        
        if target_samples != self.target_length:
            # Resample to exact target length
            resampled = np.zeros((self.target_length, num_leads), dtype=np.float32)
            for lead_idx in range(num_leads):
                resampled[:, lead_idx] = signal.resample(
                    signal_data[:, lead_idx],
                    self.target_length
                )
            return resampled
        else:
            return signal_data.astype(np.float32)
    
    def apply_highpass_filter(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Apply high-pass filter to remove baseline wander.
        
        Args:
            signal_data: ECG signal (timesteps × leads)
            
        Returns:
            Filtered signal
        """
        filtered = np.zeros_like(signal_data)
        for lead_idx in range(signal_data.shape[1]):
            filtered[:, lead_idx] = signal.filtfilt(
                self.b_high,
                self.a_high,
                signal_data[:, lead_idx]
            )
        return filtered
    
    def normalize_per_signal(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Z-score normalize each lead independently.
        
        Args:
            signal_data: ECG signal (timesteps × leads)
            
        Returns:
            Normalized signal
        """
        normalized = np.zeros_like(signal_data)
        for lead_idx in range(signal_data.shape[1]):
            lead_data = signal_data[:, lead_idx]
            mean = np.mean(lead_data)
            std = np.std(lead_data)
            if std > 1e-10:  # Avoid division by zero
                normalized[:, lead_idx] = (lead_data - mean) / std
            else:
                normalized[:, lead_idx] = lead_data - mean
        return normalized
    
    def extract_lead_ii(self, signal_data: np.ndarray) -> np.ndarray:
        """
        Extract Lead II from 12-lead ECG.
        
        Args:
            signal_data: 12-lead ECG (timesteps × 12)
            
        Returns:
            Lead II signal (1D array)
        """
        num_leads = signal_data.shape[1]
        if num_leads > self.LEAD_II_INDEX:
            return signal_data[:, self.LEAD_II_INDEX].copy()
        else:
            return signal_data[:, 0].copy()  # Fallback to first lead
    
    def preprocess(self, signal_data: np.ndarray, original_fs: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline.
        
        Steps:
        1. Resample to 500Hz, 1000 samples
        2. High-pass filter (0.5 Hz)
        3. Z-score normalize per lead
        4. Extract Lead II
        
        Args:
            signal_data: Raw ECG signal (timesteps × leads)
            original_fs: Actual sampling frequency from WFDB file (default 100 for PTB-XL lr)
        
        Returns:
            Tuple of (processed_12lead, lead_ii_signal)
            - processed_12lead: (1000, 12) normalized 12-lead signal
            - lead_ii_signal: (1000,) Lead II for clinical analysis
        """
        # 1. Resample to target length
        resampled = self.resample_to_target(signal_data, original_fs=original_fs)
        
        # 2. High-pass filter
        filtered = self.apply_highpass_filter(resampled)
        
        # 3. Normalize
        normalized = self.normalize_per_signal(filtered)
        
        # 4. Extract Lead II
        lead_ii = self.extract_lead_ii(normalized)
        
        return normalized, lead_ii
    
    def load_and_preprocess(self, filepath: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete pipeline: Load WFDB file → Preprocess.
        
        This is THE function used by both training and inference.
        
        Args:
            filepath: Path to WFDB file (without extension)
            
        Returns:
            Tuple of (processed_12lead, lead_ii_signal)
        """
        # Load raw signal with its actual sampling frequency
        raw_signal, fs = self.load_wfdb(filepath)
        
        # Preprocess using the true fs from the WFDB file
        processed_12lead, lead_ii = self.preprocess(raw_signal, original_fs=fs)
        
        return processed_12lead, lead_ii
    
    def prepare_for_model(self, signal_12lead: np.ndarray) -> np.ndarray:
        """
        Prepare signal for model input.
        
        Converts from (1000, 12) to (1, 12, 1000) for single sample.
        
        Args:
            signal_12lead: Processed 12-lead signal (1000, 12)
            
        Returns:
            Model input array (1, 12, 1000)
        """
        # Transpose to (12, 1000)
        transposed = np.transpose(signal_12lead, (1, 0))
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        return batched
    
    def load_hospital_dataset(self, hospital_dir: str) -> Tuple[List[str], np.ndarray]:
        """
        Load complete dataset for a hospital.
        
        Args:
            hospital_dir: Path to hospital directory (e.g., 'ecg_dataset/hospital_1')
            
        Returns:
            Tuple of (file_paths, labels)
            - file_paths: List of file paths (without extension) for WFDB loading
            - labels: Array of binary labels (0=Normal, 1=Arrhythmia)
        """
        hospital_path = Path(hospital_dir)
        metadata_path = hospital_path / "metadata.csv"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        # Load metadata
        df = pd.read_csv(metadata_path)
        
        # Validate required columns
        required_cols = {'filename_lr', 'label'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"metadata.csv is missing required columns: {missing_cols}")
        
        file_paths = []
        labels = []
        missing_count = 0
        
        # Shared records root: ecg_dataset/ (parent of hospital folder)
        shared_root = hospital_path.parent
        
        import logging
        logger = logging.getLogger(__name__)
        
        for _, row in df.iterrows():
            # filename_lr is like 'records100/00000/00001_lr'
            relative_path = row['filename_lr']
            
            # Try 1: path relative to hospital folder (old layout with copied files)
            full_path = str(hospital_path / relative_path)
            if os.path.exists(full_path + ".dat") and os.path.exists(full_path + ".hea"):
                file_paths.append(full_path)
                labels.append(int(row['label']))
                continue
            
            # Try 2: path relative to shared ecg_dataset root (new shared layout)
            full_path = str(shared_root / relative_path)
            if os.path.exists(full_path + ".dat") and os.path.exists(full_path + ".hea"):
                file_paths.append(full_path)
                labels.append(int(row['label']))
                continue
            
            # File genuinely missing — log warning
            missing_count += 1
            logger.warning(f"Missing WFDB file: {relative_path} (tried hospital_path and shared_root)")
        
        if missing_count > 0:
            logger.warning(f"load_hospital_dataset: {missing_count}/{len(df)} records missing from disk in {hospital_dir}")
        
        return file_paths, np.array(labels, dtype=np.int64)


# Convenience function for quick preprocessing
def preprocess_ecg(filepath: str, fs: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick preprocess function for single file.
    
    Args:
        filepath: Path to WFDB file (without extension)
        fs: Target sampling frequency
        
    Returns:
        Tuple of (processed_12lead, lead_ii_signal)
    """
    pipeline = ECGUnifiedPipeline(fs=fs)
    return pipeline.load_and_preprocess(filepath)
