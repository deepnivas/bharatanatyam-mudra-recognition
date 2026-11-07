# V3 Training & Test Feature Normalization Verification Report

**Date:** 2026-11-07 16:11:40  
**Status:** **PASS**  

## 1. Dataset Overview
- **Training Dataset (V3 Pool):** 13,416 samples (13,416 detected)
- **Official Test Set:** 1,050 samples (963 detected)

## 2. Invariance Verification Checks

| Verification Check | Training Metric | Testing Metric | Status |
|:---|:---:|:---:|:---:|
| **Schema & Feature Columns** | 63 features (`x1`..`z21`) | 63 features (`x1`..`z21`) | **PASS** |
| **Wrist Origin Placement (Landmark 1)** | Max Abs = `0.0e+00` | Max Abs = `0.0e+00` | **PASS** |
| **Scale Normalization ($\max_i \|p_i - p_0\| = 1.0$)** | Mean = `1.00000000` | Mean = `1.00000002` | **PASS** |
| **Span Range** | `[0.999999, 1.000001]` | `[0.999999, 1.000001]` | **PASS** |
| **Coordinate Bounds** | `[-1.0000, 0.9978]` | `[-0.9991, 0.8845]` | **PASS** |

## 3. Conclusion
The training dataset and the untouched official test set use mathematically identical translation anchoring ($p'_i = p_i - p_0$) and scale normalization ($p''_i = p'_i / \max \|p'_j\|$). Downstream feature engineering functions can be applied identically without coordinate domain divergence.
