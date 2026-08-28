# False Alarms: Ventricular Tachycardia Alarm Classification

This repository contains the hand-in for CM2011 Assignment 1, which explores the VTaC ventricular tachycardia alarm dataset and trains baseline classifiers to distinguish true VT alarms from false alarms.

The canonical project files are in [`hand-in/`](hand-in/). Other top-level notebooks and scripts were exploratory working files and are not needed for the final submission.

## Project Summary

Intensive care units produce many monitor alarms, and a large share of those alarms are false positives. This project uses physiological waveform data from the VTaC dataset to predict whether each ventricular tachycardia alarm is true or false.

The submitted pipeline:

1. Loads VTaC metadata and waveform records from WFDB files.
2. Maps each event to a fixed four-channel layout: `ECG1`, `ECG2`, `PLETH`, and `ABP`.
3. Handles missing signals with zero-filled channels and an availability mask.
4. Applies channel-specific filters for ECG, plethysmography, and arterial blood pressure.
5. Normalizes each available channel per event with z-score normalization.
6. Extracts simple statistical features from each channel.
7. Compares imputation strategies and classifiers.
8. Uses recursive feature elimination to inspect feature importance for the best model.

## Repository Layout

```text
hand-in/
├── fredrik_boglind_assignment1.zip          # Final submission archive
├── fredrik_boglind_assingment1.ipynb        # Main notebook
├── utility_functions.py                     # Loading, filtering, normalization, feature helpers
├── requirements.txt                         # Python dependencies
├── models/                                  # Saved local model artifacts, if present
├── raw_data/                                # Local VTaC data location; data is not committed
└── fredrik_boglind_assignment1/             # Expanded copy of the submission archive
    ├── fredrik_boglind_assignment1.py       # Python export of the notebook
    ├── fredrik_boglind_assingment1.ipynb
    ├── utility_functions(assignment1).py
    ├── requirements.txt
    └── models/
```

Note: the filename `assingment` is preserved because it is the name used in the submitted notebook.

## Data

The dataset is not included in this repository. Download VTaC from PhysioNet:

- Dataset: <https://physionet.org/content/vtac/1.0/>
- Paper: <https://proceedings.neurips.cc/paper_files/paper/2023/file/7a53bf4e02022aad32a4019d41b3b476-Paper-Datasets_and_Benchmarks.pdf>
- Reference code: <https://github.com/ML-Health/VTaC>

The notebook expects the extracted dataset at:

```text
hand-in/raw_data/vtac-a-benchmark-dataset-of-ventricular-tachycardia-alarms-from-icu-monitors-1.0/
```

That directory should contain files such as `event_labels.csv`, `benchmark_data_split.csv`, and the `waveforms/` folder.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r hand-in/requirements.txt
```

A Python 3.8 or 3.9 environment is the safest choice for the pinned package versions in `requirements.txt`. To run the notebook locally, install Jupyter if it is not already available:

```bash
python -m pip install jupyterlab
```

To run the notebook:

```bash
cd hand-in
jupyter lab fredrik_boglind_assingment1.ipynb
```

The notebook uses paths relative to `hand-in`, so running it from that directory is the simplest option.

## Method

The VTaC benchmark contains 5,037 annotated VT alarm events. Each event is a six-minute waveform segment sampled at 250 Hz, with five minutes before alarm onset and one minute after.

The final workflow loads all events into arrays of shape `(5037, 4, 90000)`, preserving every event rather than dropping rows during preprocessing. Missing channels are tracked separately and later handled through imputation at the feature level.

Feature extraction produces 20 features per event: mean, standard deviation, minimum, maximum, and range for each of the four standardized channels.

The project compares:

- Imputation: zero fill, mean, median, iterative imputation, and dropping incomplete rows.
- Models: balanced logistic regression and balanced random forest.

## Results

The best reported model is logistic regression with median imputation:

| Imputation | Model | Accuracy | F1 | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Median | Logistic regression | 0.631 | 0.588 | 0.431 | 0.927 |
| Mean | Logistic regression | 0.624 | 0.586 | 0.427 | 0.934 |
| Iterative | Logistic regression | 0.624 | 0.586 | 0.427 | 0.934 |
| Zero fill | Logistic regression | 0.629 | 0.581 | 0.428 | 0.905 |
| Drop rows | Logistic regression | 0.615 | 0.568 | 0.414 | 0.902 |
| Mean | Random forest | 0.766 | 0.519 | 0.622 | 0.445 |

Logistic regression achieved the strongest F1 score and very high recall, while random forest had higher precision but missed more true VT alarms. In this clinical setting, recall is especially important because missed true alarms are more costly than extra false positives.

Recursive feature elimination ranked standard deviation features highest, especially `ABP_std` and `PLETH_std`. This suggests that waveform variability is more informative than absolute signal level after per-event normalization.

## Repository Scope

The final submission lives in `hand-in/`. The `.gitignore` is configured so future additions stay focused on the submission directory and do not accidentally include raw waveform data, virtual environments, caches, or exploratory outputs.

If you want the Git history itself to contain only the final hand-in files, remove the already-tracked exploratory files from the index while keeping them locally:

```bash
git rm --cached 003c13_0115_plot.png bias_variance.ipynb filtering.py fredrik_boglind_assingment1*.ipynb getting_started_vtac.py load_waveforms.py requirements.txt standardize.py utility_functions.py false_alarms_assignment_description.txt
git commit -m "Limit repository to hand-in submission"
```
