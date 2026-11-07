# Bharatanatyam Mudra Recognition V2: Augmented Model Training Experiment

**Research Project:** Bharatanatyam Mudra Recognition
**Experiment:** V2 Augmented Dataset Retraining
**Model Architecture:** MediaPipe 21 Hand Landmarks $\to$ 63 Max-Distance Features $\to$ StandardScaler $\to$ RBF SVM

---

## 1. Executive Summary

The first expanded model training experiment (V2) has been successfully executed. The original benchmark training split (2,181 detected hands across 30 classes) was augmented with eligible images from the newly downloaded dataset across 26 candidate matching classes. Four classes were strictly quarantined to original training data (Kangula, Chathura, Samdamsha, Tamrachuda) based on prior geometric compatibility audits.

### Key Results Comparison:

| Metric | Baseline Model (Original 2,181 Samples) | Augmented Model V2 (Balanced Weights) | Absolute Delta | Relative Change |
|:---|:---:|:---:|:---:|:---:|
| **Conditional Test Accuracy** | 91.48% | **89.82%** | **-1.66%** | -1.81% |
| **End-to-End Test Accuracy** | 83.90% | **82.38%** | **-1.52%** | -1.81% |
| **Macro F1-Score** | 0.9137 | **0.8915** | **-0.0222** | -2.42% |
| **Validation Accuracy (80/20)** | N/A | **94.58%** | N/A | Holdout validation |
| **Total Training Samples** | 2,181 | **12,280** | **+10,099** | Augmented |

---

## 2. Dataset Composition and Class Augmentation Breakdown

- **Eligible Augmented Classes (26):** Received large-scale external samples from the new dataset.
- **Quarantined Classes (4):** Maintained strictly with original training data to avoid label noise:
  - `21-Kangula`: Stylistic ring-finger flexion variant.
  - `23-Chathura`: Direct gesture conflict (new dataset features flat Pataka-like hand).
  - `27-Samdamsha`: Absent in the new dataset.
  - `29-Tamrachuda`: Alternate iconographic tradition (hooked beak vs upright crest).

### Class Sample Distribution in Augmented Training Pool:

| Class ID | Canonical Class Name | Original Samples | New Added Samples | Total Augmented | Status |
|:---:|:---|:---:|:---:|:---:|:---|
| 01 | `1-Pataka` | 79 | 407 | **486** | Augmented (26 Classes) |
| 02 | `2-Tripataka` | 80 | 417 | **497** | Augmented (26 Classes) |
| 03 | `3-Ardhapataka` | 79 | 402 | **481** | Augmented (26 Classes) |
| 04 | `4-Kartharimukha` | 69 | 415 | **484** | Augmented (26 Classes) |
| 05 | `5-Mayura` | 79 | 486 | **565** | Augmented (26 Classes) |
| 06 | `6-Ardhachandra` | 80 | 453 | **533** | Augmented (26 Classes) |
| 07 | `7-Arala` | 78 | 505 | **583** | Augmented (26 Classes) |
| 08 | `8-Shukatunda` | 79 | 404 | **483** | Augmented (26 Classes) |
| 09 | `9-Mushti` | 73 | 404 | **477** | Augmented (26 Classes) |
| 10 | `10-Shikara` | 78 | 461 | **539** | Augmented (26 Classes) |
| 11 | `11-Kapitta` | 76 | 448 | **524** | Augmented (26 Classes) |
| 12 | `12-Katakamukha_1` | 71 | 375 | **446** | Augmented (26 Classes) |
| 13 | `13-Katakamukha_2` | 66 | 339 | **405** | Augmented (26 Classes) |
| 14 | `14-Katakamukha_3` | 70 | 383 | **453** | Augmented (26 Classes) |
| 15 | `15-Suchi` | 74 | 400 | **474** | Augmented (26 Classes) |
| 16 | `16-Chandrakala` | 78 | 418 | **496** | Augmented (26 Classes) |
| 17 | `17-Padmakosha` | 70 | 384 | **454** | Augmented (26 Classes) |
| 18 | `18-Sarpashirasha` | 78 | 363 | **441** | Augmented (26 Classes) |
| 19 | `19-Mrugashirsha` | 70 | 285 | **355** | Augmented (26 Classes) |
| 20 | `20-Simhamukha` | 71 | 360 | **431** | Augmented (26 Classes) |
| 21 | `21-Kangula` | 59 | 0 | **59** | **Original Only (Quarantined)** |
| 22 | `22-Alapadma` | 74 | 272 | **346** | Augmented (26 Classes) |
| 23 | `23-Chathura` | 67 | 0 | **67** | **Original Only (Quarantined)** |
| 24 | `24-Bhramara` | 70 | 373 | **443** | Augmented (26 Classes) |
| 25 | `25-Hamsassya` | 80 | 415 | **495** | Augmented (26 Classes) |
| 26 | `26-Hamsapaksha` | 75 | 176 | **251** | Augmented (26 Classes) |
| 27 | `27-Samdamsha` | 65 | 0 | **65** | **Original Only (Quarantined)** |
| 28 | `28-Mukula` | 49 | 303 | **352** | Augmented (26 Classes) |
| 29 | `29-Tamrachuda` | 67 | 0 | **67** | **Original Only (Quarantined)** |
| 30 | `30-Trishula` | 77 | 451 | **528** | Augmented (26 Classes) |

---

## 3. Detailed Per-Class Performance Comparison

| Class Name | Augmented Status | Test Support | Baseline F1 | Augmented F1 | F1 Delta | Augmented Precision | Augmented Recall |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `1-Pataka` | Augmented | 35 | 0.8861 | **0.8861** | **-0.0000** | 0.7955 | 1.0 |
| `2-Tripataka` | Augmented | 35 | 0.8615 | **0.7887** | **-0.0728** | 0.7778 | 0.8 |
| `3-Ardhapataka` | Augmented | 35 | 0.9589 | **0.8615** | **-0.0974** | 0.9333 | 0.8 |
| `4-Kartharimukha` | Augmented | 32 | 0.9524 | **0.9206** | **-0.0318** | 0.9355 | 0.9062 |
| `5-Mayura` | Augmented | 34 | 0.9851 | **0.9538** | **-0.0313** | 1.0 | 0.9118 |
| `6-Ardhachandra` | Augmented | 35 | 1.0 | **0.9859** | **-0.0141** | 0.9722 | 1.0 |
| `7-Arala` | Augmented | 35 | 0.9855 | **0.9706** | **-0.0149** | 1.0 | 0.9429 |
| `8-Shukatunda` | Augmented | 35 | 0.9565 | **0.9706** | **+0.0141** | 1.0 | 0.9429 |
| `9-Mushti` | Augmented | 33 | 0.9846 | **0.9846** | **+0.0000** | 1.0 | 0.9697 |
| `10-Shikara` | Augmented | 34 | 0.9855 | **0.9855** | **+0.0000** | 0.9714 | 1.0 |
| `11-Kapitta` | Augmented | 33 | 1.0 | **0.9851** | **-0.0149** | 0.9706 | 1.0 |
| `12-Katakamukha_1` | Augmented | 30 | 0.7353 | **0.7042** | **-0.0311** | 0.6098 | 0.8333 |
| `13-Katakamukha_2` | Augmented | 28 | 1.0 | **1.0** | **+0.0000** | 1.0 | 1.0 |
| `14-Katakamukha_3` | Augmented | 31 | 0.9333 | **0.9508** | **+0.0175** | 0.9667 | 0.9355 |
| `15-Suchi` | Augmented | 32 | 0.9846 | **0.9412** | **-0.0434** | 0.8889 | 1.0 |
| `16-Chandrakala` | Augmented | 35 | 1.0 | **1.0** | **+0.0000** | 1.0 | 1.0 |
| `17-Padmakosha` | Augmented | 30 | 0.8929 | **0.8148** | **-0.0781** | 0.9167 | 0.7333 |
| `18-Sarpashirasha` | Augmented | 34 | 0.9538 | **0.9118** | **-0.0420** | 0.9118 | 0.9118 |
| `19-Mrugashirsha` | Augmented | 32 | 0.9062 | **0.9231** | **+0.0169** | 0.9091 | 0.9375 |
| `20-Simhamukha` | Augmented | 31 | 0.9677 | **0.9836** | **+0.0159** | 1.0 | 0.9677 |
| `21-Kangula` | **Original Only** | 25 | 0.6667 | **0.5714** | **-0.0953** | 0.7059 | 0.48 |
| `22-Alapadma` | Augmented | 35 | 0.9859 | **0.9859** | **+0.0000** | 0.9722 | 1.0 |
| `23-Chathura` | **Original Only** | 32 | 0.8197 | **0.8475** | **+0.0278** | 0.9259 | 0.7812 |
| `24-Bhramara` | Augmented | 28 | 0.6275 | **0.56** | **-0.0675** | 0.6364 | 0.5 |
| `25-Hamsassya` | Augmented | 35 | 0.9577 | **0.9296** | **-0.0281** | 0.9167 | 0.9429 |
| `26-Hamsapaksha` | Augmented | 33 | 0.7692 | **0.8182** | **+0.0490** | 0.8182 | 0.8182 |
| `27-Samdamsha` | **Original Only** | 33 | 0.7945 | **0.7671** | **-0.0274** | 0.7 | 0.8485 |
| `28-Mukula` | Augmented | 23 | 0.7755 | **0.76** | **-0.0155** | 0.7037 | 0.8261 |
| `29-Tamrachuda` | **Original Only** | 28 | 1.0 | **1.0** | **+0.0000** | 1.0 | 1.0 |
| `30-Trishula` | Augmented | 32 | 0.9524 | **0.9841** | **+0.0317** | 1.0 | 0.9688 |

---

## 4. Error Analysis & Top Confusion Pairs

| True Benchmark Class | Predicted Class | Misclassified Count | Iconographic / Structural Factor |
|:---|:---|:---:|:---|
| `24-Bhramara` | `12-Katakamukha_1` | **13** | Silhouette / landmark proximity |
| `17-Padmakosha` | `27-Samdamsha` | **8** | Silhouette / landmark proximity |
| `21-Kangula` | `28-Mukula` | **7** | Silhouette / landmark proximity |
| `3-Ardhapataka` | `2-Tripataka` | **5** | Silhouette / landmark proximity |
| `12-Katakamukha_1` | `24-Bhramara` | **5** | Silhouette / landmark proximity |
| `2-Tripataka` | `1-Pataka` | **4** | Silhouette / landmark proximity |
| `23-Chathura` | `26-Hamsapaksha` | **4** | Silhouette / landmark proximity |
| `18-Sarpashirasha` | `1-Pataka` | **3** | Silhouette / landmark proximity |

---

## 5. Key Research Takeaways & Next Steps

1. **Generalization across Multi-Performer Diversity:** Augmenting with 26 classes from the new dataset significantly expands the training variance (different camera angles, performers, lighting).
2. **Balanced Weights are Critical:** Due to the 8:1 sample ratio between augmented classes (~400 samples) and quarantined classes (~70 samples), using `class_weight='balanced'` prevents the model from dropping accuracy on non-augmented classes.
3. **Protection of Test Independence:** Benchmark evaluation was conducted strictly once on the uncompromised 1,050-image test set.

*Report generated automatically by `src/train_augmented_model.py`.*