# Bharatanatyam Mudra Dataset Merge Compatibility Audit

**Audit Timestamp:** 2026-11-07 14:14:08  
**Elapsed Analysis Time:** 44.46 seconds  
**Original Dataset:** `C:\Users\Home\Downloads\Dance-V1.0\Kokul1984-Dance-1c60726`  
**New Dataset:** `C:\Users\Home\Bharatanatyam-Mudra-Dataset`  
**Audit Output Location:** `C:\Users\Home\bharatanatyam-mudra-recognition\reports\dataset_merge_audit`  

---

## 1. Executive Summary

This audit assesses the structural, vocabulary, and image-level compatibility between the original 30-class Bharatanatyam Mudra dataset (*Kokul et al.*) and the newly downloaded 50-class Bharatanatyam Mudra dataset (*Jisha Raj et al.*).

> [!WARNING]
> **Preliminary Finding**: The two datasets cannot be naively concatenated for model training without manual visual verification and explicit filtering. Doing so would corrupt the single-hand gesture evaluation benchmarks and introduce severe vocabulary and class-imbalance defects.

### Key Metrics Summary
| Metric | Original Dataset | New Dataset | Audit Finding |
|---|---|---|---|
| **Total Images** | 3,450 | 28,431 | Combined pool: 31,881 images |
| **Total Folders / Classes** | 30 | 50 | 20 extra folders in new dataset |
| **Single-Hand Classes** | 30 | 29 | **1 class missing in new dataset** (`27-Samdamsha`) |
| **Double-Hand Classes** | 0 | 21 | **21 double-hand classes must remain segregated** |
| **Training Split Images** | 2,400 (80/class) | Unsplit (15,396 single) | Requires stratified splitting |
| **Testing Split Images** | 1,050 (35/class) | N/A | **Original test split must remain untouched** |
| **Corrupted Images (Inventory)** | 0 | 0 | Total unreadable files: 0 |
| **Hashing Attempted / Succeeded / Failed** | 3450 / 3450 / 0 | 28431 / 28431 / 0 | Total Succeeded: 31881, Failed: 0 |
| **Cross-Dataset Exact Duplicates** | — | — | **0** exact SHA-256 matches |
| **Suspected Perceptual Duplicates** | — | — | **42** pHash matches (threshold <= 8) |
| **Potential Test-Set Leakage** | — | — | **5** matches into original Testing split |

---

## 2. Unicode Path Compatibility (Swastikam(1))

> [!NOTE]
> All 745 images in `Swastikam(1)` contain zero-width space characters (`\u200b`, code point `0x200b` / `8203`) in their filenames (e.g. `Swastika\u200bm _0.jpg`). Standard `cv2.imread()` on Windows cannot resolve paths with non-ANSI code points. The audit tool utilizes a Unicode-safe pipeline (`np.fromfile` / `read_bytes` + `cv2.imdecode` and Pillow fallback), achieving 100% successful decoding.

---

## 3. Vocabulary & Candidate Class Mapping

### Candidate Matching Matrix (30 Original Classes)
- **Candidate Matches:** 29 of 30 classes have folder name candidate matches.
- **Missing Class:** `27-Samdamsha` has **NO MATCH** in the new dataset.
- **Status:** Marked as `CANDIDATE_MATCH` and `PENDING_VISUAL_INSPECTION`. Folder name similarity does not guarantee gesture equivalence across different classical dance schools (Natya Shastra vs Abhinaya Darpana regional traditions).

### Critical Rule: The Samdamsha vs Samputa Distinction
> [!CAUTION]
> **Do not map `27-Samdamsha` to `Samputa(1)`!**
> - **Samdamsha** (*Asamyuta Hasta* / Single Hand): The thumb and index finger touch repeatedly or form a pinch/grasp gesture like a pincer.
> - **Samputa** (*Samyuta Hasta* / Double Hand): Both hands brought together with curved fingers forming an enclosed casket/box cavity.
> - Mapping Samdamsha to Samputa would introduce severe label noise and blend single-hand with double-hand recognition.

### Segregation of 21 Double-Hand (Samyuta) Classes
The new dataset contains 13,035 images across 21 double-hand mudra classes:
`Anjali(1)`, `Berunda(1)`, `Chakra(1)`, `Garuda(1)`, `Kapotham(1)`, `Karkatta(1)`, `Kartariswastika(1)`, `Katakavardhana(1)`, `Khatva(1)`, `Kilaka(1)`, `Kurma(1)`, `Matsya(1)`, `Nagabandha(1)`, `Pasha(1)`, `Pushpaputa(1)`, `Sakata(1)`, `Samputa(1)`, `Shanka(1)`, `Shivalinga(1)`, `Swastikam(1)`, `Varaha(1)`.

**Audit Action:** These 21 classes are categorized as `EXCLUDED_DOUBLE_HAND` and documented in [`class_mapping_report.csv`](class_mapping_report.csv). They must **not** be included in the 30-class single-hand recognition model.

---

## 4. Visual Contact Sheets

Contact sheets comparing 12 representative original training images against 12 new dataset images have been generated for all 30 classes under [`reports/dataset_merge_audit/contact_sheets/`](contact_sheets/).

Key observations to review in contact sheets:
1. **Pose Invariance & Variation:** Do both datasets capture the hand from similar vantage points (dorsal vs palmar view)?
2. **Wrist Positioning & Angle:** Does the new dataset include angled wrists or partial occlusions?
3. **Background & Lighting:** Original dataset is studio-shot with dark backdrops; new dataset features studio lighting from 15 volunteers.

---

## 5. Duplicate & Test-Set Leakage Analysis

- **Exact SHA-256 Duplicates:** 0
- **Perceptual Duplicates (pHash <= 8):** 42
- **Test-Set Leakage:** 5 images from the new dataset are duplicates or near-duplicates of the original Testing set.

> [!IMPORTANT]
> Any new dataset image identified in [`duplicate_report.csv`](duplicate_report.csv) with `is_test_leakage == True` **must be banned** from entering the training set. Training on images that duplicate original test samples would artificially inflate benchmark accuracy and invalidate research findings.

---

## 6. Concrete Recommendations

1. **Conduct Manual Visual Review:** Review all 30 sheets in `reports/dataset_merge_audit/contact_sheets/` to verify that candidate classes (e.g., `Katrimukha(1)`, `Kapith(1)`, `Simhamukham(1)`) are anatomically identical in hand configuration.
2. **Handle `Samdamsha` Explicitly:** Either collect additional single-hand Samdamsha images from the same volunteer setup or retain original data distributions without unbalancing class 27.
3. **Exclude Test-Set Leakage:** Filter all images listed in `duplicate_report.csv` where `is_test_leakage == True`.
4. **Isolate Double-Hand Gestures:** Keep the 21 Samyuta classes in a separate dataset branch for a dedicated future two-hand recognition pipeline.
5. **Preserve Benchmark Integrity:** Never alter the original 1,050-image test set. All evaluation comparisons must remain on the pristine test benchmark.

---

## Generated Artifacts
1. **Class Mapping Report:** [`class_mapping_report.csv`](class_mapping_report.csv)
2. **Image Inventory:** [`image_inventory.csv`](image_inventory.csv)
3. **Duplicate & Leakage Report:** [`duplicate_report.csv`](duplicate_report.csv)
4. **Corrupted Images Report:** [`corrupted_images.csv`](corrupted_images.csv)
5. **Hashing Failures Report:** [`hashing_failures.csv`](hashing_failures.csv)
6. **Image Statistics:** [`image_statistics.csv`](image_statistics.csv)
7. **Side-by-Side Visual Sheets:** [`contact_sheets/`](contact_sheets/) (30 JPEG sheets)
