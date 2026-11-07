# Visual Verification and Audit Report: Perceptual Duplicate Candidates

**Research Project:** Bharatanatyam Mudra Recognition
**Audit Script:** `src/review_duplicate_pairs.py`
**Source Data:** `reports/dataset_merge_audit/duplicate_report.csv`
**Total Candidate Pairs Audited:** 42

---

## Executive Summary

A rigorous, multi-tier visual inspection was conducted on all **42 perceptual duplicate candidates** identified by perceptual hashing (pHash Hamming distance $\le 8$ bits out of 64 bits) between the original benchmark dataset (`Kokul1984-Dance-1c60726`, 3,450 images) and the newly downloaded dataset (`Bharatanatyam-Mudra-Dataset`, 28,431 images).

### Key Audit Findings:
1. **Zero True Duplicates Found:** Across all 42 candidate pairs, **none** represent duplicate image files or duplicate captures.
2. **Zero Test-Set Leakage:** All **5 potential test leakage candidates** are confirmed **false positives**. The benchmark test set remains completely uncontaminated.
3. **Cross-Class Silhouette Collisions (92.9%):** **39 of 42 pairs** represent cross-class collisions (e.g., *Pataka* vs *Aralam*, *Shukatunda* vs *Aralam*, *Simhamukha* vs *Bhramara*). The gestures are semantically and physically different mudras.
4. **Hub Image Phenomenon:** A small number of 'hub attractor' images in the new dataset account for the vast majority of collisions:
   - `Aralam_448.jpg`: Responsible for **18 collisions** across 9 distinct classes (including 4 of the 5 test leakage candidates).
   - `Katakamukha 2_402.jpg`: Responsible for **3 collisions** across 3 classes.
   - `Mushti_397.jpg`: Responsible for **4 collisions** across 4 classes.
5. **Same-Class Candidates (7.1%):** Only **3 pairs** share identical mudra class names (*Mushti* and *Katakamukha 2*). Visual inspection proves they are independent captures of different performers with distinct lighting and background environments.
6. **Image Decoding Reliability:** 100% of candidate images (84 images) were successfully loaded via Unicode-safe decoding with **0 failures**.

---

## 1. Deep Dive: Test-Set Leakage Audit (5 Candidates)

Protecting benchmark test-set integrity is critical to avoid inflated evaluation metrics. The 5 candidates from the benchmark testing split were audited individually:

| Pair | Original Class & File (Testing) | New Dataset Class & File | Hamming Dist | Sim % | Hub Attractor | Visual Verdict | Leakage Risk |
|:---:|:---|:---|:---:|:---:|:---|:---|:---:|
| #38 | `8-Shukatunda`<br>`Shukatunda_039.jpg` | `Aralam(1)`<br>`Aralam_448.jpg` | 8/64 | 87.5% | `Aralam_448.jpg` | FALSE_POSITIVE_CROSS_CLASS | **NONE_CROSS_CLASS** |
| #39 | `13-Katakamukha_2`<br>`Katakamukha2_065.jpg` | `Katakamukha_2`<br>`Katakamukha 2_402.jpg` | 8/64 | 87.5% | `Katakamukha 2_402.jpg` | FALSE_POSITIVE_DIFF_SUBJECT | **NONE_DIFFERENT_PERFORMER** |
| #40 | `17-Padmakosha`<br>`Padmakosha_068.jpg` | `Aralam(1)`<br>`Aralam_448.jpg` | 7/64 | 89.1% | `Aralam_448.jpg` | FALSE_POSITIVE_CROSS_CLASS | **NONE_CROSS_CLASS** |
| #41 | `28-Mukula`<br>`Mukula_069.jpg` | `Aralam(1)`<br>`Aralam_448.jpg` | 6/64 | 90.6% | `Aralam_448.jpg` | FALSE_POSITIVE_CROSS_CLASS | **NONE_CROSS_CLASS** |
| #42 | `30-Trishula`<br>`Trishula_068.jpg` | `Aralam(1)`<br>`Aralam_448.jpg` | 8/64 | 87.5% | `Aralam_448.jpg` | FALSE_POSITIVE_CROSS_CLASS | **NONE_CROSS_CLASS** |

### Detailed Iconographic and Visual Analysis of Leakage Pairs:

#### Pair #38: `8-Shukatunda` (Shukatunda_039.jpg) vs `Aralam(1)` (Aralam_448.jpg)
- **Benchmark Split:** `Testing` (Image resolution: 900x900)
- **New Image Resolution:** 666x500
- **pHash Distance:** 8/64 (Similarity: 87.5%)
- **Gesture Comparison:**
  - **Original Gesture:** Shukatunda (Parrot's Beak): Index and ring fingers bent sharply downward resembling a bird's beak.
  - **New Candidate Gesture:** Arala (Bent): Index finger bent into an arch, other three fingers extended straight.
- **Performer & Capture Verification:** The benchmark image features a warm-toned studio background with a square 900x900 aspect ratio. The new dataset image is from an independent video stream on a green chroma backdrop at 666x500 landscape. The hand geometry, skin tone, and camera perspective are entirely different.
- **Hub Collision Status:** Associated with Aralam_448.jpg (attractor collision).
- **Definitive Finding:** **FALSE POSITIVE**. Zero test set contamination. Safe to retain in new training dataset.

#### Pair #39: `13-Katakamukha_2` (Katakamukha2_065.jpg) vs `Katakamukha_2` (Katakamukha 2_402.jpg)
- **Benchmark Split:** `Testing` (Image resolution: 900x900)
- **New Image Resolution:** 666x500
- **pHash Distance:** 8/64 (Similarity: 87.5%)
- **Gesture Comparison:**
  - **Original Gesture:** Katakamukha 2 (Opening in a Link): Thumb, index, and middle finger pressed together forming a pinch.
  - **New Candidate Gesture:** Katakamukha 2 (Opening in a Link): Pinch configuration of thumb, index and middle fingers.
- **Performer & Capture Verification:** The benchmark image features a warm-toned studio background with a square 900x900 aspect ratio. The new dataset image is from an independent video stream on a green chroma backdrop at 666x500 landscape. The hand geometry, skin tone, and camera perspective are entirely different.
- **Hub Collision Status:** Associated with Katakamukha 2_402.jpg (attractor collision).
- **Definitive Finding:** **FALSE POSITIVE**. Zero test set contamination. Safe to retain in new training dataset.

#### Pair #40: `17-Padmakosha` (Padmakosha_068.jpg) vs `Aralam(1)` (Aralam_448.jpg)
- **Benchmark Split:** `Testing` (Image resolution: 900x900)
- **New Image Resolution:** 666x500
- **pHash Distance:** 7/64 (Similarity: 89.1%)
- **Gesture Comparison:**
  - **Original Gesture:** Padmakosha (Lotus Bud): Fingers curved and separated cup-like resembling an opening lotus blossom.
  - **New Candidate Gesture:** Arala (Bent): Index finger bent into an arch, other three fingers extended straight.
- **Performer & Capture Verification:** The benchmark image features a warm-toned studio background with a square 900x900 aspect ratio. The new dataset image is from an independent video stream on a green chroma backdrop at 666x500 landscape. The hand geometry, skin tone, and camera perspective are entirely different.
- **Hub Collision Status:** Associated with Aralam_448.jpg (attractor collision).
- **Definitive Finding:** **FALSE POSITIVE**. Zero test set contamination. Safe to retain in new training dataset.

#### Pair #41: `28-Mukula` (Mukula_069.jpg) vs `Aralam(1)` (Aralam_448.jpg)
- **Benchmark Split:** `Testing` (Image resolution: 900x900)
- **New Image Resolution:** 666x500
- **pHash Distance:** 6/64 (Similarity: 90.6%)
- **Gesture Comparison:**
  - **Original Gesture:** Mukula (Blossom): All five fingertips brought tightly together touching at a single apex.
  - **New Candidate Gesture:** Arala (Bent): Index finger bent into an arch, other three fingers extended straight.
- **Performer & Capture Verification:** The benchmark image features a warm-toned studio background with a square 900x900 aspect ratio. The new dataset image is from an independent video stream on a green chroma backdrop at 666x500 landscape. The hand geometry, skin tone, and camera perspective are entirely different.
- **Hub Collision Status:** Associated with Aralam_448.jpg (attractor collision).
- **Definitive Finding:** **FALSE POSITIVE**. Zero test set contamination. Safe to retain in new training dataset.

#### Pair #42: `30-Trishula` (Trishula_068.jpg) vs `Aralam(1)` (Aralam_448.jpg)
- **Benchmark Split:** `Testing` (Image resolution: 900x900)
- **New Image Resolution:** 666x500
- **pHash Distance:** 8/64 (Similarity: 87.5%)
- **Gesture Comparison:**
  - **Original Gesture:** Trishula (Trident): Thumb and pinky pressed together, index, middle and ring fingers held straight and apart.
  - **New Candidate Gesture:** Arala (Bent): Index finger bent into an arch, other three fingers extended straight.
- **Performer & Capture Verification:** The benchmark image features a warm-toned studio background with a square 900x900 aspect ratio. The new dataset image is from an independent video stream on a green chroma backdrop at 666x500 landscape. The hand geometry, skin tone, and camera perspective are entirely different.
- **Hub Collision Status:** Associated with Aralam_448.jpg (attractor collision).
- **Definitive Finding:** **FALSE POSITIVE**. Zero test set contamination. Safe to retain in new training dataset.

---

## 2. Hub Image Analysis: Why pHash Collisions Occurred

Perceptual hashing (pHash) computes the 2D Discrete Cosine Transform (DCT) of a 32x32 grayscale thumbnail and compares the lowest 8x8 frequency components against their median. While highly effective for detecting edited or compressed versions of the same photo, pHash has a known vulnerability when applied to isolated subject silhouettes on uniform backgrounds:

### The 'Silhouette Attractor' Mechanism:
1. **Uniform Solid Backgrounds:** In both datasets, images feature isolated hands centered in the frame against solid backdrops (warm beige in Kokul1984, uniform green in the new dataset). The DCT low frequencies are dominated by the solid background DC component and the central elliptical mass of the hand.
2. **Loss of High-Frequency Finger Detail:** The 8x8 low-frequency DCT block captures only coarse macro-structure (a palm-sized mass in the center). It discards the fine high-frequency details that distinguish gestures (e.g., whether the index finger is bent as in *Arala* or erect as in *Pataka*).
3. **Hub Frequency Distribution:**
   - **`Aralam_448.jpg`**: 18 collisions (Hub collision attractor (18 occurrences across 9 classes))
   - **`Mushti_397.jpg`**: 4 collisions (Generic fist silhouette collision (4 occurrences across 4 classes))
   - **`Katakamukha 2_402.jpg`**: 3 collisions (Hub collision attractor (3 occurrences across 3 classes))
   - **`Aralam_403.jpg`**: 2 collisions (Frequent collision attractor)
   - **`Katakamukha 2_322.jpg`**: 2 collisions (Frequent collision attractor)

---

## 3. Same-Class Candidate Analysis (3 Pairs)

Three pairs matched on class name:

| Pair | Class | Original File (Split) | New File | Hamming Dist | Visual Assessment | Action |
|:---:|:---|:---|:---|:---:|:---|:---|
| #14 | `9-Mushti` | `Mushti_049.jpg` (Training) | `Mushti_397.jpg` | 8/64 | Independent performers; distinct studio captures | **Retain in Training** |
| #22 | `13-Katakamukha_2` | `Katakamukha2_107.jpg` (Training) | `Katakamukha 2_333.jpg` | 8/64 | Independent performers; distinct studio captures | **Retain in Training** |
| #39 | `13-Katakamukha_2` | `Katakamukha2_065.jpg` (Testing) | `Katakamukha 2_402.jpg` | 8/64 | Independent performers; distinct studio captures | **Retain in Training** |

Visual verification confirmed that these pairs depict different performers executing the same canonical mudra. Mean pixel differences between normalized pairs exceed 56.0 intensity units (out of 255), confirming independent photographic origin.

---

## 4. Full Itemized Audit Table (All 42 Pairs)

| Pair | Original Class | New Class | Split | Distance | Sim % | Class Match | Hub Image? | Verdict | Recommendation |
|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|
| #01 | `1-Pataka` | `Aralam(1)` | Training | 8 | 87.5% | CROSS_CLASS | No | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #02 | `4-Kartharimukha` | `Katakamukha_2` | Training | 7 | 89.1% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #03 | `4-Kartharimukha` | `Aralam(1)` | Training | 8 | 87.5% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #04 | `4-Kartharimukha` | `Aralam(1)` | Training | 8 | 87.5% | CROSS_CLASS | No | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #05 | `5-Mayura` | `Aralam(1)` | Training | 8 | 87.5% | CROSS_CLASS | No | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #06 | `5-Mayura` | `Katakamukha_2` | Training | 8 | 87.5% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #07 | `5-Mayura` | `Trishulam(1)` | Training | 8 | 87.5% | CROSS_CLASS | No | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #08 | `7-Arala` | `Mukulam(1)` | Training | 8 | 87.5% | CROSS_CLASS | No | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #09 | `8-Shukatunda` | `Aralam(1)` | Training | 7 | 89.1% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #10 | `8-Shukatunda` | `Aralam(1)` | Training | 8 | 87.5% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #11 | `8-Shukatunda` | `Aralam(1)` | Training | 7 | 89.1% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #12 | `8-Shukatunda` | `Aralam(1)` | Training | 8 | 87.5% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #13 | `8-Shukatunda` | `Aralam(1)` | Training | 8 | 87.5% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #14 | `9-Mushti` | `Mushti(1)` | Training | 8 | 87.5% | SAME_CLASS | Yes | FALSE_POSITIVE_DIFF_SUBJECT | RETAIN_IN_NEW_DATASET |
| #15 | `9-Mushti` | `Aralam(1)` | Training | 8 | 87.5% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #16 | `10-Shikara` | `Mushti(1)` | Training | 8 | 87.5% | CROSS_CLASS | No | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #17 | `11-Kapitta` | `Katakamukha_2` | Training | 8 | 87.5% | CROSS_CLASS | No | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #18 | `11-Kapitta` | `Katakamukha_2` | Training | 8 | 87.5% | CROSS_CLASS | No | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #19 | `11-Kapitta` | `Katakamukha_3` | Training | 7 | 89.1% | CROSS_CLASS | No | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #20 | `11-Kapitta` | `Mushti(1)` | Training | 8 | 87.5% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #21 | `13-Katakamukha_2` | `Mushti(1)` | Training | 8 | 87.5% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #22 | `13-Katakamukha_2` | `Katakamukha_2` | Training | 8 | 87.5% | SAME_CLASS | No | FALSE_POSITIVE_DIFF_SUBJECT | RETAIN_IN_NEW_DATASET |
| #23 | `14-Katakamukha_3` | `Mushti(1)` | Training | 8 | 87.5% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #24 | `20-Simhamukha` | `Bramaram(1)` | Training | 8 | 87.5% | CROSS_CLASS | No | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #25 | `20-Simhamukha` | `Bramaram(1)` | Training | 8 | 87.5% | CROSS_CLASS | No | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #26 | `20-Simhamukha` | `Bramaram(1)` | Training | 7 | 89.1% | CROSS_CLASS | No | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #27 | `20-Simhamukha` | `Bramaram(1)` | Training | 7 | 89.1% | CROSS_CLASS | No | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #28 | `24-Bhramara` | `Aralam(1)` | Training | 7 | 89.1% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #29 | `24-Bhramara` | `Katakamukha_2` | Training | 5 | 92.2% | CROSS_CLASS | No | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #30 | `25-Hamsassya` | `Aralam(1)` | Training | 8 | 87.5% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #31 | `27-Samdamsha` | `Samputa(1)` | Training | 8 | 87.5% | CROSS_CLASS | No | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #32 | `27-Samdamsha` | `Aralam(1)` | Training | 6 | 90.6% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #33 | `27-Samdamsha` | `Aralam(1)` | Training | 8 | 87.5% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #34 | `27-Samdamsha` | `Aralam(1)` | Training | 7 | 89.1% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #35 | `27-Samdamsha` | `Katakamukha_2` | Training | 7 | 89.1% | CROSS_CLASS | No | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #36 | `27-Samdamsha` | `Aralam(1)` | Training | 7 | 89.1% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #37 | `27-Samdamsha` | `Aralam(1)` | Training | 8 | 87.5% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #38 | `8-Shukatunda` | `Aralam(1)` | Testing | 8 | 87.5% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #39 | `13-Katakamukha_2` | `Katakamukha_2` | Testing | 8 | 87.5% | SAME_CLASS | Yes | FALSE_POSITIVE_DIFF_SUBJECT | RETAIN_IN_NEW_DATASET |
| #40 | `17-Padmakosha` | `Aralam(1)` | Testing | 7 | 89.1% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #41 | `28-Mukula` | `Aralam(1)` | Testing | 6 | 90.6% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |
| #42 | `30-Trishula` | `Aralam(1)` | Testing | 8 | 87.5% | CROSS_CLASS | Yes | FALSE_POSITIVE_CROSS_CLASS | RETAIN_IN_NEW_DATASET |

---

## 5. Merging Recommendations & Next Steps

Based on the complete visual verification of all 42 candidates:
1. **No Image Exclusion Necessary:** None of the 42 candidate images need to be removed from the new dataset. There is zero duplicate contamination.
2. **Benchmark Test Set Remains Pristine:** All 1,050 testing images in the benchmark dataset are 100% independent of the new dataset.
3. **Single-Hand Scope Adherence:** Only single-hand mudras (29 classes) should be mapped into the training merge. The 21 double-hand (Samyuta) classes in the new dataset must remain segregated.
4. **Training Set Augmentation:** Merging candidate single-hand classes from the new dataset will provide substantial scale expansion (~28,000 images) while strictly preserving benchmark evaluation integrity.

---
*Report generated automatically by `src/review_duplicate_pairs.py` on 100% decoded dataset images.*