# RAMs: Recursive Attention Mechanisms for Multimodal Prognostic Prediction in NPC

## 📌 Overview
This repository contains the core source code for the **Multimodal Recursive Attention Mechanisms (RAMs) Fusion Network**, as described in our manuscript on the prognostic risk stratification for non-metastatic Nasopharyngeal Carcinoma (NPC). 

By integrating macroscopic MRI radiomics, clinical baseline factors, and microscopic spatial pathomics (derived from H&E whole-slide images), this multimodal framework overcomes the limitations of traditional TNM staging, enabling precise individualized prediction of Progression-Free Survival (PFS).

## ⚠️ Data Privacy Statement
Due to institutional data security regulations and strict patient privacy protection (HIPAA compliance), the full clinical and multimodal imaging/pathology datasets used in our retrospective cohort (n=618) cannot be publicly shared. 

However, to demonstrate the operability of the analytical pipeline, we have provided a minimal, synthetic dummy dataset (`dummy_cohort.csv`). **The provided code is fully functional and can be seamlessly applied to your own formatted datasets.**

## 📁 Repository Structure
* **`01_core_algorithms.py`**: Contains the mathematical logic for feature engineering.
  * **Part 1:** Extracts spatial topological features (Spatial Shannon Entropy [SSE], Tumor-Immune Interaction Index [TIII]) based on Delaunay triangulation of 2D cellular coordinates.
  * **Part 2:** Implements a strict nested 10-fold cross-validation Logistic-LASSO algorithm for robust feature dimensionality reduction.
* **`02_RAMs_Architecture.py`**: The core deep learning architecture built with PyTorch. It defines the hierarchical cross-attention layers, aligning radiomics anchors with clinical factors, and subsequently recursively reweighting with spatial pathomics vectors.
* **`03_Clinical_Evaluation.R`**: The gold-standard clinical statistics pipeline written in R. It covers the computation of C-index, DCA (Decision Curve Analysis), Continuous NRI/IDI, and the construction of the visual predictive Nomogram.
* **`requirements.txt`**: List of dependencies for the Python environment.
* **`dummy_cohort.csv`**: A synthetic dataset containing 10 mock patients to facilitate code testing.

## 🛠️ System Requirements
This codebase has been tested on **Python 3.8+** and **R 4.2.1+**. 

To install the required Python dependencies, run:
`pip install -r requirements.txt`

*For the R scripts, please ensure you have installed the following packages from CRAN: `survival`, `rms`, `dcurves`, `survIDINRI`, and `timeROC`.*

## 🚀 How to Use (Sanity Check)
We have included built-in test modules in the Python scripts to verify the environment setup. 

To test the feature extraction and LASSO selection:
`python 01_core_algorithms.py`

To test the PyTorch RAMs forward pass and tensor dimensions:
`python 02_RAMs_Architecture.py`

## 📝 Citation
If you find this code or our methodology useful in your research, please cite our corresponding paper published in the *Journal of Translational Medicine*. (Citation details will be updated upon publication).

---
*For any technical questions regarding the code implementation, please feel free to open an issue or contact the corresponding authors.*