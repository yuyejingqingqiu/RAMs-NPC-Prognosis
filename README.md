# RAMs: Recursive Attention Mechanisms for Multimodal Prognostic Prediction in NPC
 
## Overview
This repository provides representative code for a multimodal recursive attention mechanisms (RAMs) prognostic model for non-metastatic nasopharyngeal carcinoma (NPC).
 
The demonstration pipeline includes spatial pathomics feature calculation, representative feature selection, a PyTorch implementation of the RAMs fusion architecture, and clinical evaluation scripts using a synthetic demonstration dataset. The repository is intended to support methodological transparency and code demonstration. It is not intended to reproduce the original patient-level results because the original clinical, radiological, and pathological data cannot be publicly shared.
 
## Data Privacy Statement
Due to patient privacy, ethical approval restrictions, and institutional data governance requirements, the original clinical, radiological, and pathological datasets from the retrospective cohort cannot be made publicly available.
 
A minimal synthetic demonstration dataset (`dummy_cohort.csv`) is provided only for code testing and workflow illustration. It does not contain real patient data and should not be used to reproduce the reported cohort-level results.
 
## Repository Structure
- `01_core_algorithms.py`: Representative code for spatial topology feature calculation, including spatial Shannon entropy (SSE), tumor-immune interaction index (TIII), and nearest immune distance (NID), as well as a demonstration Logistic-LASSO feature selection pipeline.
- `02_RAMs_Architecture.py`: PyTorch implementation of the representative RAMs fusion network.
- `03_Clinical_Evaluation.R`: Representative clinical evaluation script, including Cox modeling, time-dependent ROC analysis, DeLong comparison, calibration, decision curve analysis, NRI/IDI, and nomogram construction.
- `04_SHAP_Interpretation.py`: Demonstration script for SHAP-based model interpretation using synthetic tabular features.
- `dummy_cohort.csv`: Synthetic demonstration dataset containing mock patients for code testing.
- `requirements.txt`: Python dependencies for the demonstration scripts.
 
## System Requirements
Python 3.8+ and R 4.2.1+ are recommended.
 
Install Python dependencies:
 
```bash
pip install -r requirements.txt
```
 
For the R script, please install the following R packages:
 
```r
install.packages(c("survival", "rms", "dcurves", "survIDINRI", "timeROC", "pROC", "ResourceSelection"))
```
 
## How to Run the Demonstration
Test spatial feature calculation and LASSO selection:
 
```bash
python 01_core_algorithms.py
```
 
Test the RAMs forward pass:
 
```bash
python 02_RAMs_Architecture.py
```
 
Run clinical evaluation with the synthetic demonstration dataset:
 
```r
source("03_Clinical_Evaluation.R")
```
 
Run the SHAP demonstration:
 
```bash
python 04_SHAP_Interpretation.py
```
 
## Citation
Citation information will be updated upon publication. For now, please cite this repository and the associated archived version if using the demonstration code.
 
## Contact
For technical questions regarding the demonstration code, please contact the corresponding authors or open an issue in this repository.
