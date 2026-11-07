# RUNBOOK.md — Environment, Build, Test & Run Execution Guide

## 1. Environment Setup

### Prerequisites
- Python 3.11+
- pip

### Clean Virtual Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,visualization]"
```

## 2. Running Verification & Tests

### Run Full Test Suite
```powershell
pytest -v
```

### Run Unit Tests Only
```powershell
pytest tests/unit -v
```

### Run Integration Tests Only
```powershell
pytest tests/integration -v
```

### Run Adversarial & Regression Suite
```powershell
pytest tests/regression -v
```

## 3. CLI Commands

### 1-Command Deterministic Offline Demo
```powershell
python -m volsurfacelab demo --source synthetic --seed 42 --output-dir outputs/demo
```

### Calibrate Custom European Quotes CSV
```powershell
python -m volsurfacelab calibrate --input data/samples/synthetic_quotes.csv --config configs/synthetic_default.yaml --output-dir outputs/run_01
```

### Verify Saved Calibration Artifact
```powershell
python -m volsurfacelab verify --artifact outputs/demo/calibration.json
```

### Generate Standalone Report
```powershell
python -m volsurfacelab report --artifact outputs/demo/calibration.json --output-dir outputs/demo
```

### Run Model Ablation Benchmark
```powershell
python -m volsurfacelab benchmark --config configs/synthetic_default.yaml --output-dir outputs/benchmark
```

### Launch Interactive Web Product
```powershell
python -m volsurfacelab dashboard
# Or directly via streamlit:
streamlit run src/volsurfacelab/dashboard/app.py
```
To launch with a specific pre-computed artifact:
```powershell
python -m volsurfacelab dashboard --artifact outputs/demo/run_artifact.json --port 8501
```
