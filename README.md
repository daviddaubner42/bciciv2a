# BCICIV-2a: Motor-Imagery EEG — Artifact Correction & CSP–LDA Decoding

A reproducible Python/MNE pipeline for preprocessing motor-imagery EEG and decoding imagined movements with Common Spatial Patterns (CSP) + Linear Discriminant Analysis (LDA).

The main focus of this repository is a systematic comparison of blink-artifact correction strategies on the BCI Competition IV 2a dataset:

 1. EOG regression (subtractive, EOG-channel driven)

 2. ICA + find_bads_eog (component rejection based on EOG correlation)

 3. ICA + ICLabel (automatic component classification)

Each variant is saved separately so downstream decoding accuracy can be compared across preprocessing choices.

### Highlights

    * End-to-end MNE-based pipeline for the BCI Competition IV 2a dataset.

    * Blink correction compared across three methods (EOG regression, ICA + EOG correlation, ICA + ICLabel).

    * Uses automatic artifact-component selection (no manual component inspection) so the pipeline scales to larger datasets.

    * Decoding via CSP + LDA with cross-validation for preprocessing method selection and a held-out test-set evaluation for final score.

### Dataset

This project uses the BCI Competition IV 2a dataset.

    * Source: https://www.bbci.de/competition/iv/#download

    * Task: 4-class motor imagery — left hand, right hand, both feet, tongue.

    * Subjects: 9 (A01 … A09).

    * Sessions: 2 per subject — one training session (T) and one evaluation session (E).

    * Channels: 22 EEG + 3 monopolar EOG (EOG-left, EOG-central, EOG-right).

    * Sampling rate: 250 Hz.

## Obtaining the data

The dataset is **not redistributed in this repository** — it is subject to the competition's own terms of use. You must download it yourself from the [BBCI site](https://www.bbci.de/competition/iv/#download) and accept their data-use agreement.

After downloading, extract the archive so that the GDF files sit in a directory named `BCICIV_2a_gdf/` at the repository root:

```text
BCICIV_2a_gdf/
├── A01T.gdf
├── A01E.gdf
├── A02T.gdf
├── A02E.gdf
├── ...
├── A09T.gdf
└── A09E.gdf
```

## Repository structure

```text
.
├── preprocessing_evaluation.ipynb   # Main notebook: preprocessing, artifact correction, decoding
├── requirements.txt                 # Requirements file
├── BCICIV_2a_gdf/                   # Dataset (not tracked; download separately)
│   ├── A01T.gdf ... A09T.gdf        # Training sessions
│   └── A01E.gdf ... A09E.gdf        # Evaluation sessions
│
├── raw/                             # Created by the notebook — raw recordings as MNE .fif
│   └── A01_raw.fif ... A09_raw.fif
├── eog_regression/                  # Created by the notebook — fitted EOG-regression weights
│   └── A01_regr_weights.pkl ... A09_regr_weights.pkl
├── ica/                             # Created by the notebook — fitted ICA solutions
│   ├── A01_ica.fif,     A01_iclabel.fif
│   └── ...
└── preproc/                         # Created by the notebook — corrected/processed signals
    ├── A01_regr_raw.fif             # After EOG regression
    ├── A01_ica_raw.fif              # After ICA + find_bads_eog
    └── A01_iclabel_raw.fif          # After ICA + ICLabel
```

### Requirements

All requirements are in the `requirements.txt` file in repository root.

### Installation

```bash
pip install -r requirements.txt
```

### Usage

1. Download the dataset and place the .gdf files in BCICIV_2a_gdf/ (see Dataset).

2. Launch the notebook:
```bash
jupyter notebook preprocessing_evaluation.ipynb
```

3. Run all cells. The notebook will:

    * Read each A0XT.gdf (or reuse the cached raw/A0X_raw.fif).

    * Mark EOG channels and apply CAR.

    * Fit and apply EOG regression; save weights.

    * Apply a 2 Hz high-pass filter.

    * Fit ICA (15 components) and reject blink components via find_bads_eog and ICLabel separately.

    * Save each preprocessing variant to preproc/.

4. Inspect the diagnostic plots (before/after EOG regression, before/after high-pass filtering).

| The notebook caches aggressively: if raw/A0X_raw.fif already exists it is loaded instead of re-parsing the GDF. Delete the relevant folder to force a rebuild.

### Outputs
| Path | Contents |
|------|----------|
| `raw/A0X_raw.fif` | Raw GDF recording, cached as MNE Raw |
| `eog_regression/A0X_regr_weights.pkl` | Fitted EOGRegression object (reusable on E sessions) |
| `preproc/A0X_regr_raw.fif` | Signal after CAR + EOG regression |
| `ica/A0X_ica.fif` | ICA solution, find_bads_eog variant |
| `ica/A0X_iclabel.fif` | ICA solution, ICLabel variant |
| `preproc/A0X_ica_raw.fif` | Signal after CAR + ICA (EOG correlation) correction |
| `preproc/A0X_iclabel_raw.fif` | Signal after CAR + ICA (ICLabel) correction |

### References

* Dataset: Brunner, C., Leeb, R., Müller-Putz, G., Schlögl, A., & Pfurtscheller, G. (2008). BCI Competition 2008 – Graz data set A. https://www.bbci.de/competition/iv/desc_2a.pdf

* High-pass filtering before ICA: Winkler, I., Debener, S., Müller, K.-R., & Tangermann, M. (2015). On the influence of high-pass filtering on ICA-based artifact reduction in EEG-ERP. EMBC. https://pubmed.ncbi.nlm.nih.gov/26737196/

* ICLabel: Pion-Tonachini, L., Kreutz-Delgado, K., & Makeig, S. (2019). ICLabel: An automated electroencephalographic independent component classifier, dataset, and website. NeuroImage.

* MNE-Python: Gramfort, A. et al. (2013). MEG and EEG data analysis with MNE-Python. Frontiers in Neuroscience.

* CSP: Ramoser, H., Müller-Gerking, J., & Pfurtscheller, G. (2000). Optimal spatial filtering of single trial EEG during imagined hand movement. IEEE Trans. Rehabil. Eng.

### License and citation

The code in this repository is released under the MIT License.

The BCI Competition IV 2a dataset is not covered by this license and remains subject to the terms of use set by the BBCI / Graz University of Technology. Do not redistribute the .gdf files.