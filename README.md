# Federated Learning System — Heart Disease + ECG Arrhythmia

A privacy-preserving **Federated Learning** web application with two prediction modules:
1. **Heart Disease Risk** — tabular data (BRFSS 2022), FedAvg aggregation
2. **ECG Arrhythmia Detection** — 12-lead ECG signals (PTB-XL), ResNet1D-18 + FedBN + Hybrid Clinical/ML decision system

Raw patient data never leaves each hospital node — only model weights are shared.

---

## Tech Stack

| Layer       | Technology                                              |
|-------------|--------------------------------------------------------|
| Backend     | Flask 3, Flask-Login, Flask-Migrate, SQLAlchemy, SQLite |
| ML/DL       | PyTorch 2 (ResNet1D-18), scikit-learn, scipy           |
| ECG Data    | PTB-XL via WFDB library                                |
| Frontend    | Custom dark UI, HTML5 Canvas (ECG visualization)       |

---

## Project Structure

```
Federated_Learning/
├── app/
│   ├── api/routes.py           # All REST API endpoints
│   ├── auth/                   # Login / register
│   ├── fl/
│   │   ├── ecg_model.py        # ResNet1D-18 model
│   │   ├── ecg_client.py       # Hospital FL client
│   │   ├── ecg_aggregator.py   # FedBN aggregator
│   │   ├── ecg_clinical.py     # Pan-Tompkins + Hybrid decision
│   │   ├── ecg_unified.py      # Preprocessing pipeline
│   │   ├── aggregator.py       # Heart disease FedAvg aggregator
│   │   ├── client.py           # Heart disease FL client
│   │   ├── model.py            # Heart disease DNN
│   │   └── data.py             # Heart disease data loader
│   ├── main/                   # Page routes
│   ├── templates/              # Jinja2 HTML templates
│   ├── models.py               # DB models
│   └── fl_globals.py           # Global aggregator instances
├── ecg_dataset/                # PTB-XL metadata CSVs + split_hospitals.py
│   ├── hospital_1/metadata.csv
│   ├── hospital_2/metadata.csv
│   ├── hospital_3/metadata.csv
│   └── split_hospitals.py      # Run once to create hospital splits
├── heart_disease_dataset/      # BRFSS 2022 hospital splits
├── scripts/seed.py             # Creates DB users/roles/hospitals
├── migrations/                 # Alembic DB migrations
├── config.py
├── run.py
└── requirements.txt
```

---

## Setup on a New Machine

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Clone & install
```bash
git clone <repo-url>
cd Federated_Learning
pip install -r requirements.txt
```

### 3. Download PTB-XL dataset (required for ECG module)
```bash
# Download from PhysioNet — about 1.8 GB
# https://physionet.org/content/ptb-xl/1.0.3/
# Extract so the structure looks like:
#   ecg_dataset/records100/00000/00001_lr.dat
#   ecg_dataset/records100/00000/00001_lr.hea
#   ecg_dataset/ptbxl_database.csv
#   ecg_dataset/scp_statements.csv
```

### 4. Generate hospital dataset splits
```bash
cd ecg_dataset
python split_hospitals.py
# Creates: hospital_1/metadata.csv, hospital_2/metadata.csv, hospital_3/metadata.csv
cd ..
```

### 5. Set up the database
```bash
flask db upgrade          # Apply migrations
python scripts/seed.py    # Create roles, hospitals, users
```

### 6. Run the server
```bash
set PYTHONPATH=%CD%        # Windows
# export PYTHONPATH=$PWD   # Linux/Mac
python run.py
# OR:
flask run --debug --no-reload
```
Open: http://127.0.0.1:5000

---

## Default Login Credentials

| Role     | Username    | Password       | Access                          |
|----------|-------------|----------------|---------------------------------|
| Admin    | `admin`     | `admin123`     | User mgmt, FL rounds, aggregation |
| Doctor   | `doctor`    | `doctor123`    | Heart disease & ECG prediction  |
| Hospital | `hospital1` | `hospital123`  | ECG local training (Node 1)     |
| Hospital | `hospital2` | `hospital2123` | ECG local training (Node 2)     |
| Hospital | `hospital3` | `hospital3123` | ECG local training (Node 3)     |

---

## ECG Federated Learning Workflow

1. Log in as each **Hospital** node → go to ECG Training → Train on Local Dataset
2. After all 3 hospitals train → log in as **Admin** → Aggregate ECG Models (FedBN)
3. Log in as **Doctor** → ECG tab → upload `.dat + .hea` files → Analyze ECG
4. Result shows: Rhythm (Normal/Arrhythmia), Confidence, RR Variability, Heart Rate, R-Peaks, Lead II waveform with attention heatmap

## Heart Disease FL Workflow

1. Admin starts FL round
2. Hospital nodes upload BRFSS CSV splits and train local DNN models
3. Admin aggregates (FedAvg) → global model updates
4. Doctor uses global model for cardiovascular risk prediction

---

## What Works Out of the Box (After Setup)

| Feature | Status |
|---------|--------|
| Login / RBAC (Admin, Doctor, Hospital) | ✅ Works immediately after seed |
| Heart Disease prediction (pre-trained) | ✅ If hospital CSVs present |
| ECG prediction (clinical rules only) | ✅ Works without training — uses RR-CV thresholds |
| ECG prediction (ML model) | ⚠️ Requires training all 3 hospitals first |
| Lead II rhythm visualization | ✅ Shown after every ECG analysis |
| Federated aggregation | ✅ Admin dashboard |

> **Note:** The ECG clinical override (CV > 20% → Arrhythmia, CV < 5% → Normal) works immediately even without any ML training. For borderline cases the ML model needs to be trained first.
