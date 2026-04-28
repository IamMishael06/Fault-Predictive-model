# 🛢️ Oil Well Fault Detection — Predictive Maintenance with Machine Learning

> Detecting operational faults in oil and gas wells before they occur, using real-world sensor data from the 3W Dataset.

---

## Overview

This project builds a multiclass fault detection system for offshore oil wells. Using time-series sensor data recorded at 1-second intervals from permanent downhole and topside instrumentation, the goal is to classify well operating states — distinguishing normal operation from 8 distinct fault types and their early transient warning phases.

The project emphasises **early fault detection** — identifying the transient warning period before a fault fully develops, which is where predictive maintenance delivers the most value.

---

## Dataset

**Source:** [3W Dataset — Petrobras / UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/540/3w+dataset)

The 3W dataset contains time-series recordings from real oil wells (WELL), physics-based simulations (SIMULATED), and domain-expert hand-drawn scenarios (DRAWN).

### Fault Classes

| Class | Description |
|-------|-------------|
| 0 | Normal Operation |
| 1 | Abrupt Increase of BSW |
| 2 | Incipient Leak in Production Choke |
| 3 | Flow Instability |
| 4 | Rapid Productivity Loss |
| 5 | Quick Restriction in Production Choke |
| 6 | Scaling in Production Choke |
| 7 | Hydrate in Production Line |
| 8 | Hydrate in Service Line |
| 10X | Transient period leading into Fault X (e.g. 101 = early warning for Fault 1) |

---

## Sensors

| Column | Description | Unit |
|--------|-------------|------|
| P-PDG | Pressure — Permanent Downhole Gauge | bar |
| P-TPT | Pressure — Topside Transducer | bar |
| T-TPT | Temperature — Topside Transducer | °C |
| P-MON-CKP | Pressure upstream of Production Choke | bar |
| T-JUS-CKP | Temperature downstream of Production Choke | °C |
| P-JUS-CKGL | Pressure downstream of Gas Lift Choke | bar |
| QGL | Gas Lift Flow Rate | m³/s |
| gas_lift_active | Binary flag — whether gas lift was active | 0/1 |

---

## Project Structure

```
├── data/                        # Raw 3W dataset (download separately)
│   ├── 0/                       # Normal operation episodes
│   ├── 1/                       # Fault class 1 episodes
│   └── .../
├── resampled_csv/               # Processed per-class CSVs
│   ├── 0/compressed_0.csv
│   └── .../
├── notebooks/
│   └── 3wdataset.ipynb          # Main analysis and modelling notebook
├── cleaned_dataset.csv          # Final cleaned training dataset
├── training_set1.csv            # Training split
├── test_set1.csv                # Test split
└── README.md
```

---

## Methodology

### 1. Data Processing
- Resampled raw 1-second data to **1-minute intervals** using mean aggregation
- Handled missing sensors: dropped P-PDG (corrupted) and T-JUS-CKGL (not installed)
- Forward-filled T-TPT sensor failures within each well episode
- Filled gas lift columns (P-JUS-CKGL, QGL) with 0 where gas lift was inactive
- Created binary `gas_lift_active` feature from QGL availability

### 2. Feature Engineering
- Rolling window statistics (mean, std, variance) per sensor over 10-minute windows
- Cyclical month encoding (sin/cos) to capture seasonal effects on hydrate formation
- Pressure unit conversion from Pascals to bar

### 3. Train/Test Split
- Temporal split — training data taken from the chronological first 75% of each class
- Test data from the remaining 25%, ensuring no future data leaks into training
- Capped at 40,000 rows per class to reduce imbalance

### 4. Models Trained
- `RandomForestClassifier`
- `HistGradientBoostingClassifier` ← best performer
- `LogisticRegression`

---

## Results

| Model | Accuracy |
|-------|----------|
| RandomForestClassifier | 0.519 |
| HistGradientBoostingClassifier | 0.633 |
| LogisticRegression | 0.655 |

> Note: Accuracy is a misleading metric here due to class imbalance. Macro recall is the primary evaluation metric — the cost of missing a real fault far outweighs the cost of a false alarm.

---

## Key Challenges

- **Class imbalance** — Normal operation (class 0) has 80x more samples than rare faults like class 7
- **Sensor corruption** — P-PDG had systematic zero readings spanning multiple days due to downhole sensor failures
- **Missing sensors** — T-JUS-CKGL not installed on any well in this dataset
- **Transient class ambiguity** — Classes 101–108 look nearly identical to class 0 in raw sensor readings, making early detection inherently difficult

---

## Requirements

```
pandas
numpy
scikit-learn
matplotlib
```

---

## Getting Started

```bash
git clone https://github.com/yourusername/oil-well-fault-detection
cd oil-well-fault-detection

# Download the 3W dataset from the link above
# Extract into the data/ folder

pip install -r requirements.txt
jupyter notebook notebooks/3wdataset.ipynb
```

---

## References

- Vargas, R. et al. (2019). *A realistic and public dataset with rare undesirable real events in oil wells.* Journal of Petroleum Science and Engineering.
- [3W Dataset GitHub](https://github.com/ricardovvargas/3w_dataset)
- [UCI ML Repository — 3W Dataset](https://archive.ics.uci.edu/dataset/540/3w+dataset)
