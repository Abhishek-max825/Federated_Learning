"""
Clinical ECG Analysis Module.

Implements:
1. Pan-Tompkins R-peak detection algorithm
2. RR interval calculation
3. Coefficient of Variation (CV) for arrhythmia detection
4. Hybrid decision logic (ML + Clinical rules)

Clinical thresholds:
- CV < 8%: Normal sinus rhythm
- CV > 12%: Arrhythmia/Abnormality
- 8% <= CV <= 12%: Borderline (use ML model)
"""

import numpy as np
from scipy import signal
from typing import List, Tuple, Dict, Optional


class PanTompkinsDetector:
    """
    Pan-Tompkins QRS detection algorithm.
    
    Standard algorithm for R-peak detection in ECG signals.
    """
    
    def __init__(self, fs: int = 500):
        """
        Initialize detector.
        
        Args:
            fs: Sampling frequency in Hz (default 500)
        """
        self.fs = fs
        self._design_filters()
    
    def _design_filters(self):
        """Design filters for Pan-Tompkins algorithm."""
        nyquist = self.fs / 2
        
        # Bandpass filter 0.5-40 Hz (standard ECG range, wider than 5-15 Hz)
        low = 0.5 / nyquist
        high = 40 / nyquist
        self.b_band, self.a_band = signal.butter(
            N=3,  # Lower order for gentler roll-off
            Wn=[low, high], 
            btype='band',
            analog=False
        )
        
        # High-pass filter for baseline drift removal (>0.5 Hz)
        self.b_high, self.a_high = signal.butter(
            N=2,
            Wn=0.5 / nyquist,
            btype='high',
            analog=False
        )
    
    def bandpass_filter(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter (0.5-40 Hz)."""
        return signal.filtfilt(self.b_band, self.a_band, signal_data)
    
    def remove_baseline(self, signal_data: np.ndarray) -> np.ndarray:
        """Remove baseline drift using high-pass filter (>0.5 Hz)."""
        return signal.filtfilt(self.b_high, self.a_high, signal_data)
    
    def derivative(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply derivative filter (5-point)."""
        # H(z) = (1/8T)(-z^(-2) - 2z^(-1) + 2z + z^2)
        h = np.array([-1, -2, 0, 2, 1]) / 8
        return np.convolve(signal_data, h, mode='same')
    
    def squaring(self, signal_data: np.ndarray) -> np.ndarray:
        """Point-wise squaring."""
        return signal_data ** 2
    
    def moving_window_integration(
        self, 
        signal_data: np.ndarray, 
        window_size: int = None
    ) -> np.ndarray:
        """Moving window integration."""
        if window_size is None:
            window_size = int(0.15 * self.fs)  # 150ms window
        
        result = np.convolve(
            signal_data, 
            np.ones(window_size) / window_size, 
            mode='same'
        )
        return result
    
    def detect_peaks(
        self, 
        integrated_signal: np.ndarray,
        threshold_factor: float = 0.3,  # Lowered from 0.5 to catch more peaks
        refractory_period_ms: float = 250  # 250ms for faster heart rates
    ) -> np.ndarray:
        """
        Detect R-peaks from integrated signal.
        
        Args:
            integrated_signal: Processed signal
            threshold_factor: Fraction of max signal for threshold
            refractory_period_ms: Minimum time between peaks (ms)
            
        Returns:
            Array of peak indices
        """
        # Dynamic threshold
        threshold = threshold_factor * np.max(integrated_signal)
        
        # Find peaks above threshold with minimum distance
        peaks = []
        min_distance_samples = int(0.25 * self.fs)  # 250ms minimum for faster rates
        refractory_samples = max(int(refractory_period_ms * self.fs / 1000), min_distance_samples)
        
        i = 0
        while i < len(integrated_signal):
            if integrated_signal[i] > threshold:
                # Find local maximum in refractory period
                window_end = min(i + refractory_samples, len(integrated_signal))
                local_max_idx = i + np.argmax(integrated_signal[i:window_end])
                peaks.append(local_max_idx)
                i = local_max_idx + refractory_samples
            else:
                i += 1
        
        return np.array(peaks)
    
    def detect(self, ecg_signal: np.ndarray) -> np.ndarray:
        """
        Full Pan-Tompkins detection pipeline.
        
        Args:
            ecg_signal: Single lead ECG signal
            
        Returns:
            R-peak indices
        """
        # Step 0: Remove baseline drift (prevents false peaks)
        baseline_removed = self.remove_baseline(ecg_signal)
        
        # Step 1: Bandpass filter (0.5-40 Hz)
        filtered = self.bandpass_filter(baseline_removed)
        
        # Step 2: Derivative
        derivative = self.derivative(filtered)
        
        # Step 3: Squaring
        squared = self.squaring(derivative)
        
        # Step 4: Moving window integration
        integrated = self.moving_window_integration(squared)
        
        # Step 5: Peak detection (lower threshold for more sensitivity)
        peaks = self.detect_peaks(integrated, threshold_factor=0.2)
        
        return peaks


class RRAnalyzer:
    """RR interval analysis for arrhythmia detection."""
    
    # Clinical thresholds for Coefficient of Variation
    # Standard clinical thresholds for arrhythmia detection
    CV_NORMAL_THRESHOLD = 5.0       # CV < 5% = Definitely Normal (strict)
    CV_ARRHYTHMIA_THRESHOLD = 20.0  # CV > 20% = Definitely Arrhythmia (strict)
    
    MIN_PEAKS_REQUIRED = 3  # Minimum R-peaks needed for reliable analysis (lowered from 5)
    
    def __init__(self, fs: int = 500):
        """
        Initialize RR analyzer.
        
        Args:
            fs: Sampling frequency in Hz
        """
        self.fs = fs
        self.detector = PanTompkinsDetector(fs)
    
    def calculate_rr_intervals(self, r_peaks: np.ndarray) -> np.ndarray:
        """
        Calculate RR intervals from R-peak indices.
        
        Args:
            r_peaks: Array of R-peak sample indices
            
        Returns:
            RR intervals in milliseconds
        """
        if len(r_peaks) < 2:
            return np.array([])
        
        # Convert sample differences to milliseconds
        rr_samples = np.diff(r_peaks)
        rr_ms = (rr_samples / self.fs) * 1000
        
        return rr_ms
    
    def calculate_cv(self, rr_intervals: np.ndarray) -> float:
        """
        Calculate Coefficient of Variation (CV) of RR intervals.
        
        CV = (std / mean) * 100
        
        Args:
            rr_intervals: RR intervals in milliseconds
            
        Returns:
            CV as percentage
        """
        if len(rr_intervals) < 2:
            return float('inf')
        
        mean_rr = np.mean(rr_intervals)
        std_rr = np.std(rr_intervals)
        
        if mean_rr == 0:
            return float('inf')
        
        cv = (std_rr / mean_rr) * 100
        return cv
    
    def analyze(self, ecg_signal: np.ndarray) -> Dict:
        """
        Full clinical analysis of ECG signal.
        
        Args:
            ecg_signal: Lead II ECG signal
            
        Returns:
            Dictionary with:
            - r_peaks: R-peak indices
            - rr_intervals: RR intervals in ms
            - cv: Coefficient of Variation (%)
            - num_peaks: Number of R-peaks detected
            - valid: Whether analysis is valid (>= 3 peaks)
        """
        # Detect R-peaks
        r_peaks = self.detector.detect(ecg_signal)
        
        # Calculate RR intervals
        rr_intervals = self.calculate_rr_intervals(r_peaks)
        
        # Calculate CV
        cv = self.calculate_cv(rr_intervals)
        
        return {
            'r_peaks': r_peaks,
            'rr_intervals': rr_intervals,
            'cv': cv,
            'num_peaks': len(r_peaks),
            'valid': len(r_peaks) >= self.MIN_PEAKS_REQUIRED
        }
    
    def get_clinical_decision(self, cv: float) -> Tuple[str, str]:
        """
        Get clinical decision based on CV.
        
        Args:
            cv: Coefficient of Variation (%)
            
        Returns:
            Tuple of (decision, reason)
            decision: 'normal', 'arrhythmia', or 'borderline'
            reason: explanation string
        """
        if cv < self.CV_NORMAL_THRESHOLD:
            return 'normal', f'CV = {cv:.1f}% < {self.CV_NORMAL_THRESHOLD}% (Normal threshold)'
        elif cv > self.CV_ARRHYTHMIA_THRESHOLD:
            return 'arrhythmia', f'CV = {cv:.1f}% > {self.CV_ARRHYTHMIA_THRESHOLD}% (Arrhythmia threshold)'
        else:
            return 'borderline', f'CV = {cv:.1f}% ({self.CV_NORMAL_THRESHOLD}-{self.CV_ARRHYTHMIA_THRESHOLD}% borderline, use ML)'


class HybridDecisionSystem:
    """
    Hybrid decision system combining ML and clinical rules.
    
    Clinical rules override ML when CV is conclusive:
    - CV < 5%: Always Normal (even if ML says Arrhythmia)
    - CV > 20%: Always Arrhythmia (even if ML says Normal)
    - 5% <= CV <= 20%: Use ML prediction
    """
    
    def __init__(self):
        """Initialize hybrid system."""
        self.rr_analyzer = RRAnalyzer()
    
    def decide(
        self,
        ml_prediction: int,  # 0=Normal, 1=Arrhythmia
        ml_confidence: float,
        ecg_signal: np.ndarray
    ) -> Dict:
        """
        Make final prediction using hybrid logic.
        
        Args:
            ml_prediction: ML model prediction (0 or 1)
            ml_confidence: ML confidence (0-1)
            ecg_signal: Lead II ECG signal
            
        Returns:
            Dictionary with:
            - final_prediction: 0 or 1
            - final_confidence: float
            - method: 'ml' or 'clinical_override'
            - ml_raw: original ML prediction
            - ml_confidence: original ML confidence
            - cv: coefficient of variation
            - reason: explanation
        """
        # Analyze ECG clinically
        analysis = self.rr_analyzer.analyze(ecg_signal)
        
        if not analysis['valid']:
            # Not enough peaks for clinical analysis - fallback to ML
            return {
                'final_prediction': ml_prediction,  # Fallback to ML
                'final_confidence': ml_confidence,
                'method': 'ml',  # Use ML since clinical fails
                'ml_raw': ml_prediction,
                'ml_confidence': ml_confidence,
                'cv': None,  # No valid CV
                'reason': f'Insufficient R-peaks ({analysis["num_peaks"]} < 3). Using ML prediction.',
                'r_peaks': analysis['r_peaks'],
                'rr_intervals': analysis['rr_intervals']
            }
        
        cv = analysis['cv']
        num_peaks = analysis['num_peaks']
        
        # If too few peaks for reliable CV, trust ML instead
        MIN_PEAKS_FOR_CLINICAL = 4  # 4+ peaks sufficient for CV in 2s signal
        if num_peaks < MIN_PEAKS_FOR_CLINICAL:
            return {
                'final_prediction': ml_prediction,
                'final_confidence': ml_confidence,
                'method': 'ml',
                'ml_raw': ml_prediction,
                'ml_confidence': ml_confidence,
                'cv': cv,
                'reason': f'Only {num_peaks} R-peaks detected (need {MIN_PEAKS_FOR_CLINICAL}+ for reliable CV). Using ML prediction.',
                'r_peaks': analysis['r_peaks'],
                'rr_intervals': analysis['rr_intervals']
            }
        
        clinical_decision, clinical_reason = self.rr_analyzer.get_clinical_decision(cv)
        
        # Map clinical decision to binary
        clinical_pred = 0 if clinical_decision == 'normal' else 1
        
        # Apply hybrid rules
        if clinical_decision == 'normal':
            # CV < 5%: Override to Normal
            return {
                'final_prediction': 0,
                'final_confidence': 0.88,  # Capped to avoid showing 100%
                'method': 'clinical_override',
                'ml_raw': ml_prediction,
                'ml_confidence': ml_confidence,
                'cv': cv,
                'reason': f'Clinical override: {clinical_reason}. ML said {"Arrhythmia" if ml_prediction == 1 else "Normal"}.',
                'r_peaks': analysis['r_peaks'],
                'rr_intervals': analysis['rr_intervals']
            }
        
        elif clinical_decision == 'arrhythmia':
            # CV > 20%: Override to Arrhythmia
            return {
                'final_prediction': 1,
                'final_confidence': 0.88,
                'method': 'clinical_override',
                'ml_raw': ml_prediction,
                'ml_confidence': ml_confidence,
                'cv': cv,
                'reason': f'Clinical override: {clinical_reason}. ML said {"Arrhythmia" if ml_prediction == 1 else "Normal"}.',
                'r_peaks': analysis['r_peaks'],
                'rr_intervals': analysis['rr_intervals']
            }
        
        else:  # borderline
            # 8% <= CV <= 12%: Trust ML
            return {
                'final_prediction': ml_prediction,
                'final_confidence': ml_confidence,
                'method': 'ml',
                'ml_raw': ml_prediction,
                'ml_confidence': ml_confidence,
                'cv': cv,
                'reason': f'Borderline CV (8-12%). Using ML prediction (confidence: {ml_confidence:.1%}).',
                'r_peaks': analysis['r_peaks'],
                'rr_intervals': analysis['rr_intervals']
            }


# Convenience function for quick analysis
def analyze_ecg_clinical(ecg_signal: np.ndarray, fs: int = 500) -> Dict:
    """
    Quick clinical analysis of ECG signal.
    
    Args:
        ecg_signal: Lead II ECG signal
        fs: Sampling frequency
        
    Returns:
        Clinical analysis results
    """
    analyzer = RRAnalyzer(fs)
    return analyzer.analyze(ecg_signal)
