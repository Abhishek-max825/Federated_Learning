# ECG Federated Learning System — Full Technical Documentation

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Dataset — PTB-XL](#dataset--ptb-xl)
4. [Dataset Split — Hospital Partitioning](#dataset-split--hospital-partitioning)
5. [Preprocessing Pipeline](#preprocessing-pipeline)
6. [Model — ResNet1D-18](#model--resnet1d-18)
7. [Federated Learning Flow](#federated-learning-flow)
8. [Hospital Training (ECG FL Client)](#hospital-training-ecg-fl-client)
9. [Server Aggregation (FedBN)](#server-aggregation-fedbn)
10. [Differential Privacy](#differential-privacy)
11. [Clinical Analysis — Pan-Tompkins + RR CV](#clinical-analysis--pan-tompkins--rr-cv)
12. [Hybrid Decision System](#hybrid-decision-system)
13. [Prediction API — Doctor Node](#prediction-api--doctor-node)
14. [Training API — Hospital Node](#training-api--hospital-node)
15. [PTB-XL Label Mapping](#ptb-xl-label-mapping)
16. [File Structure](#file-structure)
17. [Known Issues and Fixes](#known-issues-and-fixes)

---

## Overview

This system implements **Federated Learning for ECG arrhythmia detection** across 3 hospital nodes. No patient data is ever shared — only model weight updates are sent to the central server.

**Task**: Binary classification — `Normal` vs `Arrhythmia`  
**Data**: PTB-XL dataset (21,837 clinical 12-lead ECG records)  
**Model**: ResNet1D-18 (1D Residual CNN for time-series)  
**Aggregation**: FedBN (Federated Batch Normalization)  
**Clinical check**: Pan-Tompkins R-peak detection + RR Coefficient of Variation

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        CENTRAL SERVER                                │
│                                                                      │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │                   FedBN Aggregator                          │   │
│   │  Global Model (ResNet1D-18, binary, 2 classes)             │   │
│   │  FedAvg weights (skip BatchNorm running stats)             │   │
│   └──────────────────────┬──────────────────────────────────────┘   │
│              sends global weights ↓  ↑ receives local weight deltas  │
└──────────────────────────┼──────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  HOSPITAL 1  │  │  HOSPITAL 2  │  │  HOSPITAL 3  │
│              │  │              │  │              │
│  Local Data  │  │  Local Data  │  │  Local Data  │
│  (private)   │  │  (private)   │  │  (private)   │
│              │  │              │  │              │
│  ECGFLClient │  │  ECGFLClient │  │  ECGFLClient │
│  trains on   │  │  trains on   │  │  trains on   │
│  local WFDB  │  │  local WFDB  │  │  local WFDB  │
└──────────────┘  └──────────────┘  └──────────────┘

                    ↓ Doctor uploads .dat + .hea

              ┌───────────────────────┐
              │   DOCTOR NODE         │
              │                       │
              │  Upload ECG file      │
              │  → Preprocess         │
              │  → ResNet1D predict   │
              │  → Pan-Tompkins RR    │
              │  → Hybrid decision    │
              │  → Show result        │
              └───────────────────────┘
```

---

## Dataset — PTB-XL

| Property | Value |
|----------|-------|
| **Source** | PhysioNet (Wagner et al., 2020) |
| **Records** | 21,837 clinical 12-lead ECGs |
| **Patients** | 18,885 |
| **Duration** | 10 seconds per record |
| **Sampling rates** | 100 Hz (records100/) and 500 Hz (records500/) |
| **Format** | WFDB — `.dat` (signal) + `.hea` (header/metadata) |
| **Labels** | SCP-ECG standard diagnostic codes |
| **Sex split** | Male 52%, Female 48% |
| **Used rate** | 100 Hz (`records100/`) for training |

### Label File

`ecg_dataset/ptbxl_database.csv` contains:

| Column | Description |
|--------|-------------|
| `ecg_id` | Unique record ID (matches filename: `00001_lr`) |
| `scp_codes` | Dict of diagnostic codes, e.g., `{'NORM': 100.0}` |
| `filename_lr` | Relative path to 100Hz WFDB file |
| `label` | Binary label — 0=Normal, 1=Arrhythmia |

---

## Dataset Split — Hospital Partitioning

**Script**: `ecg_dataset/split_hospitals.py`

Splits PTB-XL into 3 hospital datasets for FL simulation.

### How It Works

```
ptbxl_database.csv
       ↓
Parse scp_codes → filter to NORM or ARRHYTHMIA
       ↓
Assign binary label (0 or 1)
       ↓
Split by ecg_id into 3 equal thirds
       ↓
hospital_1/  hospital_2/  hospital_3/
```

### SCP Code Mapping

| Code | Meaning | Label |
|------|---------|-------|
| `NORM` | Normal Sinus Rhythm | 0 |
| `AFIB` | Atrial Fibrillation | 1 |
| `AFLT` | Atrial Flutter | 1 |
| `SVES` | Supraventricular Ectopic Beat | 1 |
| `VES` | Ventricular Ectopic Beat | 1 |
| `SBRAD` | Sinus Bradycardia | 1 |
| `TACHY` | Tachycardia | 1 |
| `SARRH` | Sinus Arrhythmia | 1 |

### Output Structure

```
ecg_dataset/
├── hospital_1/
│   ├── metadata.csv          ← Labels + file paths for Hospital 1
│   └── records100/
│       └── 00000/
│           ├── 00001_lr.dat
│           └── 00001_lr.hea
├── hospital_2/
│   ├── metadata.csv
│   └── records100/
└── hospital_3/
    ├── metadata.csv
    └── records100/
```

### Run Split

```bash
cd ecg_dataset
python split_hospitals.py
```

---

## Preprocessing Pipeline

**File**: `app/fl/ecg_unified.py` — `ECGUnifiedPipeline`

**Same pipeline is used for BOTH training and inference** (prevents distribution mismatch).

### Steps

```
WFDB file (.dat + .hea)
       ↓
wfdb.rdsamp() → (n_samples, 12) raw signal
       ↓
Resample to 1000 timesteps (scipy.signal.resample)
       ↓
Normalize per lead: (x - mean) / (std + 1e-8)
       ↓
Clip to [-10, 10] (artifact removal)
       ↓
Shape: (12, 1000)  ← 12 leads, 1000 timesteps
       ↓
Lead II extracted separately for clinical analysis
```

### Key Method

```python
ecg_12lead, lead_ii = pipeline.load_and_preprocess(wfdb_path)
# ecg_12lead: (12, 1000) numpy array
# lead_ii: (1000,) numpy array — used for R-peak detection
```

---

## Model — ResNet1D-18

**File**: `app/fl/ecg_model.py`

A **1D ResNet-18** architecture adapted for time-series ECG signals.

### Architecture

```
Input: (batch, 12, 1000)  ← 12 leads, 1000 timesteps
   ↓
Conv1D(12→64, k=15, stride=2) + BN + ReLU + MaxPool
   ↓
Layer1: 2× BasicBlock1D(64→64)
   ↓
Layer2: 2× BasicBlock1D(64→128, stride=2)
   ↓
Layer3: 2× BasicBlock1D(128→256, stride=2)
   ↓
Layer4: 2× BasicBlock1D(256→512, stride=2)
   ↓
Attention Conv1D(512→1) → softmax weights
   ↓
AdaptiveAvgPool1d(1) → (batch, 512)
   ↓
FC(512→2) → [Normal, Arrhythmia] logits
   ↓
Softmax → probabilities
```

### BasicBlock1D

Each residual block:
```
input
  ├── Conv1D(k=7) → BN → ReLU → Conv1D(k=7) → BN
  └── (downsample if shape changes)
       ↓ add
     ReLU → output
```

### Attention Mechanism

Generates per-timestep importance weights:
- `attention_conv`: `Conv1D(512, 1, k=1)` → attention scores
- Applied before pooling to weight important ECG segments
- Output used for visualization (heatmap in UI)

### Key Methods

```python
model = ECGModel(num_classes=2, in_channels=12, attention=True)
weights = model.get_weights()          # Returns state_dict
model.set_weights(weights)             # Loads state_dict
predictions, probs, attn = model.predict_with_attention(X)
metrics = model.train_local(dataloader, epochs=5, lr=0.001)
```

---

## Federated Learning Flow

### Full Round

```
1. HOSPITAL sends training request to server
          ↓
2. SERVER provides current global model weights
          ↓
3. HOSPITAL trains locally (5 epochs, batch=16)
   - Data never leaves hospital
          ↓
4. HOSPITAL applies Differential Privacy:
   - Clip weight deltas (max_norm=5.0)
   - Add Gaussian noise (multiplier=0.01)
          ↓
5. HOSPITAL sends weight update to server
          ↓
6. SERVER aggregates updates (FedBN)
   - Weighted average by sample count
   - Skip BatchNorm running stats
          ↓
7. SERVER updates global model
          ↓
8. All hospitals get improved model on next prediction
```

---

## Hospital Training (ECG FL Client)

**File**: `app/fl/ecg_client.py` — `ECGFLClient`

### Initialization

```python
client = ECGFLClient(
    client_id=1,                            # Hospital ID
    hospital_dir='ecg_dataset/hospital_1'  # Local data path
)
# Loads: metadata.csv → file_paths + labels
# Prints: "ECG Client 1: Loaded N local samples | Normal: X, Arrhythmia: Y"
```

### Dataset Loading

`ECGLocalDataset` loads WFDB files **on-demand** (lazy loading):
- `__getitem__`: loads one WFDB file, preprocesses via unified pipeline
- Returns `(tensor(12, 1000), label)` per sample

### Training

```python
trained_weights, n_samples, metrics = client.train(
    global_weights=global_weights,
    epochs=5,
    batch_size=16,
    lr=0.001
)
# metrics = {'accuracy': float, 'loss': float}
```

- Batch size auto-adjusted if dataset smaller than 16
- Validates dataset is non-empty before training

---

## Server Aggregation (FedBN)

**File**: `app/fl/ecg_aggregator.py` — `FedBNAggregator`

### FedBN vs FedAvg

| Parameter Type | FedAvg | FedBN |
|---------------|--------|-------|
| Conv weights | ✅ Aggregate | ✅ Aggregate |
| FC weights | ✅ Aggregate | ✅ Aggregate |
| BN `weight`, `bias` | ✅ Aggregate | ✅ Aggregate |
| BN `running_mean` | ✅ Aggregate | ❌ Keep local |
| BN `running_var` | ✅ Aggregate | ❌ Keep local |
| BN `num_batches_tracked` | ✅ Aggregate | ❌ Keep local |

**Why FedBN?** Hospitals have different patient populations (non-IID data). Each hospital's BN statistics reflect their local distribution — aggregating them hurts accuracy.

### Aggregation Formula

```
w_global[k] = Σ (n_i / N) * w_i[k]    for all non-BN params
# where n_i = samples from hospital i, N = total samples
```

### Security Checks

- **Shape validation**: rejects weights with wrong tensor shapes
- **NaN/Inf detection**: rejects corrupt updates
- **Weight clipping**: L2 norm capped at `MAX_WEIGHT_NORM = 50.0`

---

## Differential Privacy

**File**: `app/fl/ecg_client.py` — inside `train()` method

Protects against membership inference attacks.

### Implementation

```python
for each parameter k:
    delta = trained_weights[k] - pre_train_weights[k]
    
    # 1. Clip delta (L2 sensitivity bound)
    delta_norm = ||delta||_2
    if delta_norm > max_delta_norm (5.0):
        delta = delta * (5.0 / delta_norm)
    
    # 2. Add Gaussian noise
    noise = randn_like(delta.float()) * (0.01 * 5.0)
    delta = delta + noise.to(delta.dtype)  # cast back to original dtype
    
    trained_weights[k] = pre_train_weights[k] + delta
```

### Parameters

| Parameter | Value | Effect |
|-----------|-------|--------|
| `max_delta_norm` | 5.0 | Max change per parameter tensor |
| `noise_multiplier` | 0.01 | Noise scale (low = less noise = better accuracy) |

> **Note**: `noise.to(delta.dtype)` is required — `torch.randn_like` doesn't support Long tensors.

---

## Clinical Analysis — Pan-Tompkins + RR CV

**File**: `app/fl/ecg_clinical.py`

Clinical validation layer that runs alongside ML prediction.

### Pan-Tompkins R-Peak Detection

```
Lead II signal (1000 samples @ 100Hz)
       ↓
Step 0: High-pass filter (>0.5 Hz) → remove baseline drift
       ↓
Step 1: Bandpass filter (0.5–40 Hz) → isolate QRS range
       ↓
Step 2: 5-point derivative → highlight sharp edges (R-peaks)
       ↓
Step 3: Squaring → emphasize high-frequency peaks
       ↓
Step 4: Moving window integration (150ms window) → smooth
       ↓
Step 5: Peak detection (threshold=0.2×max, refractory=250ms)
       ↓
R-peak indices array
```

### RR Interval Calculation

```
RR intervals (ms) = diff(r_peak_indices) * (1000 / sampling_rate)
CV = (std(RR) / mean(RR)) * 100    ← Coefficient of Variation in %
```

### Clinical Thresholds

| CV Value | Interpretation |
|----------|---------------|
| CV < 8% | Regular rhythm → **Normal** |
| CV 8–12% | Borderline → **Use ML** |
| CV > 12% | Irregular rhythm → **Arrhythmia** |

### Reliability Requirements

| R-Peaks Found | Action |
|---------------|--------|
| < 3 | Invalid — fallback to ML |
| 3–9 | Too few for reliable CV — fallback to ML |
| ≥ 10 | Reliable — clinical override allowed |

---

## Hybrid Decision System

**File**: `app/fl/ecg_clinical.py` — `HybridDecisionSystem`

Combines ML prediction with clinical RR analysis.

### Decision Logic

```
Run Pan-Tompkins on Lead II
       ↓
Count R-peaks detected
       │
       ├─ < 3 peaks → FALLBACK: Use ML prediction as-is
       │
       ├─ 3–9 peaks → FALLBACK: Use ML prediction as-is
       │                (too few for reliable clinical CV)
       │
       └─ ≥ 10 peaks → Calculate CV
                             │
                             ├─ CV < 8%  → CLINICAL OVERRIDE: Normal (conf=95%)
                             ├─ CV > 12% → CLINICAL OVERRIDE: Arrhythmia (conf=95%)
                             └─ 8–12%    → USE ML prediction + confidence
```

### Output Dictionary

```python
{
    'final_prediction': 0 or 1,       # 0=Normal, 1=Arrhythmia
    'final_confidence': float,         # 0.0–1.0
    'method': 'ml' or 'clinical_override',
    'ml_raw': int,                     # original ML prediction
    'ml_confidence': float,
    'cv': float or None,               # RR CV% (None if insufficient peaks)
    'reason': str,                     # Human-readable explanation
    'r_peaks': np.ndarray,
    'rr_intervals': np.ndarray
}
```

---

## Prediction API — Doctor Node

**Endpoint**: `POST /api/ecg/predict`  
**File**: `app/api/routes.py` — `ecg_predict()`

### Request

Upload `.dat` + `.hea` files as `multipart/form-data`:

```javascript
const formData = new FormData();
formData.append('file', datFile);   // 00001_lr.dat
formData.append('file', heaFile);   // 00001_lr.hea

fetch('/api/ecg/predict', {
    method: 'POST',
    credentials: 'same-origin',
    body: formData
});
```

### Server Flow

```
Receive .dat + .hea files
       ↓
Save to temp directory
       ↓
load_and_preprocess(wfdb_path) → (ecg_12lead, lead_ii)
       ↓
pipeline.prepare_for_model() → X shape (1, 12, 1000)
       ↓
global_model.predict_with_attention(X)
→ predictions, probabilities, attention_maps
       ↓
HybridDecisionSystem.decide(ml_pred, ml_conf, lead_ii)
→ final_prediction, method, cv, reason
       ↓
Calculate RR variability, heart rate, R-peak count
       ↓
Save to ECGUpload + ECGPrediction database tables
       ↓
Return JSON response
```

### Response

```json
{
    "prediction": "Normal",
    "confidence": 0.9823,
    "method": "ml",
    "rr_variability": 5.2,
    "heart_rate": 72,
    "r_peaks": 14,
    "decision_basis": "ml",
    "clinical_reason": "CV = 5.2% < 8% (Normal threshold)",
    "ml_prediction": "Normal",
    "ml_confidence": 0.9823,
    "clinical_override": false,
    "attention_map": [0.002, 0.003, ...],
    "debug_info": {
        "rr_intervals_ms": [820.0, 830.0, ...],
        "mean_rr_ms": 825.0,
        "cv_percent": 5.2
    }
}
```

### JSON Safety

- All `float('inf')` values are converted to `null` before serialization
- `raw_class` uses `ml_prediction` (not undefined variable)

---

## Training API — Hospital Node

**Endpoint**: `POST /api/ecg/train`  
**File**: `app/api/routes.py` — `ecg_train_local()`

### Two Training Modes

#### Mode 1: Train on Local Dataset

Uses pre-split hospital data with correct PTB-XL labels.

```
Hospital ID from session
       ↓
Load ecg_dataset/hospital_{id}/
       ↓
ECGFLClient trains for 5 epochs
       ↓
FedBN aggregator receives weight update
```

#### Mode 2: Train on Uploaded Files

Upload `.dat` + `.hea` files, labels looked up from PTB-XL database.

```
Upload .dat + .hea files
       ↓
Save to temp/records100/
       ↓
For each .dat file:
    Extract ecg_id from filename (e.g. "00001_lr" → 1)
    Lookup scp_codes in ptbxl_database.csv
    get_label_from_scp_codes() → 0 or 1
       ↓
Create metadata.csv with correct labels
       ↓
ECGFLClient trains on temp directory
       ↓
Logs: "Labels — Normal: X, Arrhythmia: Y"
```

---

## PTB-XL Label Mapping

**Function**: `get_label_from_scp_codes()` in `app/api/routes.py`

```python
scp_codes = "{'NORM': 100.0, 'SR': 0.0}"
# → parse dict
# → find primary code (highest confidence)
# → check against arrhythmia code list
# → return 0 (Normal) or 1 (Arrhythmia)
```

### Normal Codes → Label 0

`NORM`, `SR` (Sinus Rhythm), `SBRAD` (Sinus Bradycardia)

### Arrhythmia Codes → Label 1

`AFIB`, `AFLT`, `SARRH`, `IMI`, `ASMI`, `AMI`, `ABQRS`, `STACH`, `SVARR`, `LNGQT`, `PVC`, `PAC`

**Logic**: If any arrhythmia code has confidence > 50%, label = 1. Otherwise, if primary code is normal, label = 0.

---

## File Structure

```
Federated_Learning/
├── ecg_dataset/
│   ├── ptbxl_database.csv         ← Full PTB-XL metadata + SCP codes
│   ├── split_hospitals.py          ← Script to split data into hospitals
│   ├── records100/                 ← All 100Hz WFDB files
│   ├── hospital_1/
│   │   ├── metadata.csv            ← Labels for Hospital 1
│   │   └── records100/             ← Hospital 1's WFDB files
│   ├── hospital_2/
│   │   ├── metadata.csv
│   │   └── records100/
│   └── hospital_3/
│       ├── metadata.csv
│       └── records100/
│
├── app/
│   ├── fl/
│   │   ├── ecg_model.py            ← ResNet1D-18 architecture
│   │   ├── ecg_client.py           ← Hospital FL client
│   │   ├── ecg_aggregator.py       ← FedBN aggregation server
│   │   ├── ecg_unified.py          ← Unified preprocessing pipeline
│   │   └── ecg_clinical.py         ← Pan-Tompkins + RR CV analysis
│   │
│   ├── api/
│   │   └── routes.py               ← /ecg/train and /ecg/predict endpoints
│   │
│   └── templates/
│       ├── hospital/
│       │   └── dashboard.html      ← Hospital training UI
│       └── doctor/
│           └── dashboard.html      ← Doctor prediction UI
│
└── ECG_README.md                   ← This file
```

---

## Known Issues and Fixes

### 1. `NotImplementedError: "normal_kernel_cpu" not implemented for 'Long'`

**Cause**: `torch.randn_like()` called on a Long tensor in DP noise generation.  
**Fix**: Convert to float first, then cast noise back.
```python
noise = torch.randn_like(delta.float()) * (noise_multiplier * max_delta_norm)
delta = delta + noise.to(delta.dtype)
```

### 2. `cv: Infinity` breaks JSON parsing

**Cause**: Python's `float('inf')` is not valid JSON.  
**Fix**: All infinity values replaced with `None` (JSON `null`) before `jsonify()`.

### 3. 100% Training Accuracy

**Cause**: All uploaded files were assigned `label: 0` (Normal).  
**Fix**: PTB-XL database lookup now assigns correct labels via `get_label_from_scp_codes()`.

### 4. `num_samples=0` error on "Train on Local Dataset"

**Cause**: Double `records100/` in path — `filename_lr` already includes `records100/` but code appended it again.  
**Fix**: Changed `records_dir / relative_path` to `hospital_path / relative_path`.

### 5. Wrong predictions (Normal ECGs show Arrhythmia)

**Cause**: Pan-Tompkins detected only 5–6 R-peaks (should be 10–15), causing wrong CV.  
**Fix**:
- Widened bandpass filter: 5–15 Hz → 0.5–40 Hz
- Added baseline drift removal (high-pass > 0.5 Hz)
- Lowered peak threshold: 0.5 → 0.2
- Added fallback: < 10 R-peaks → use ML instead of clinical override

### 6. Flask server crashes during training

**Cause**: Flask auto-reloader restarts server when Python files change.  
**Fix**: Start server with `use_reloader=False`:
```bash
set PYTHONPATH=%CD% && python -c "from app import create_app; app = create_app(); app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)"
```
