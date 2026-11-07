# Two-Stage Detection Cascade Evaluation Report (V3)

- **Dataset Evaluated:** C:\Users\Home\Bharatanatyam-Mudra-Dataset (26 classes, 13,844 images)
- **Stage 1:** Raw Image, MediaPipe Confidence 0.5
- **Stage 2:** 1.5x Linear Upscale, MediaPipe Confidence 0.3 (triggered only upon Stage 1 failure)

## Overall Cascade Performance

- **Total Images:** 13,844
- **Stage 1 Baseline Detections:** 10,099 (72.95%)
- **Stage 2 Recovered Images:** 1,136 (Recovered 30.33% of Stage 1 misses)
- **Total Cascade Detections:** **11,235** (**81.15%**)
- **Images Lost from Stage 1:** **0** (Stage 1 detections strictly preserved)
- **Net Gain Over Baseline:** **+1,136 images**
- **Invalid Landmarks:** **0**
- **Total Processing Time:** 416.3 seconds (33.3 FPS)

## Per-Class Breakdown Table

| Class Name | Total Images | Stage 1 (Conf 0.5) | Stage 1 Rate | Stage 2 Recovered | Cascade Total | Cascade Rate | Gain (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `1-Pataka` | 568 | 407 | 71.7% | +29 | **436** | **76.8%** | **+5.1%** |
| `2-Tripataka` | 551 | 417 | 75.7% | +39 | **456** | **82.8%** | **+7.1%** |
| `3-Ardhapataka` | 577 | 402 | 69.7% | +31 | **433** | **75.0%** | **+5.4%** |
| `4-Kartharimukha` | 564 | 415 | 73.6% | +69 | **484** | **85.8%** | **+12.2%** |
| `5-Mayura` | 534 | 486 | 91.0% | +28 | **514** | **96.3%** | **+5.2%** |
| `6-Ardhachandra` | 463 | 453 | 97.8% | +3 | **456** | **98.5%** | **+0.6%** |
| `7-Arala` | 615 | 505 | 82.1% | +52 | **557** | **90.6%** | **+8.5%** |
| `8-Shukatunda` | 559 | 404 | 72.3% | +52 | **456** | **81.6%** | **+9.3%** |
| `9-Mushti` | 490 | 404 | 82.4% | +29 | **433** | **88.4%** | **+5.9%** |
| `10-Shikara` | 530 | 461 | 87.0% | +26 | **487** | **91.9%** | **+4.9%** |
| `11-Kapitta` | 558 | 448 | 80.3% | +52 | **500** | **89.6%** | **+9.3%** |
| `12-Katakamukha_1` | 492 | 375 | 76.2% | +58 | **433** | **88.0%** | **+11.8%** |
| `13-Katakamukha_2` | 488 | 339 | 69.5% | +48 | **387** | **79.3%** | **+9.8%** |
| `14-Katakamukha_3` | 507 | 383 | 75.5% | +45 | **428** | **84.4%** | **+8.9%** |
| `15-Suchi` | 485 | 400 | 82.5% | +23 | **423** | **87.2%** | **+4.7%** |
| `16-Chandrakala` | 466 | 418 | 89.7% | +14 | **432** | **92.7%** | **+3.0%** |
| `17-Padmakosha` | 495 | 384 | 77.6% | +59 | **443** | **89.5%** | **+11.9%** |
| `18-Sarpashirasha` | 563 | 363 | 64.5% | +37 | **400** | **71.0%** | **+6.6%** |
| `19-Mrugashirsha` | 559 | 285 | 51.0% | +48 | **333** | **59.6%** | **+8.6%** |
| `20-Simhamukha` | 523 | 360 | 68.8% | +45 | **405** | **77.4%** | **+8.6%** |
| `22-Alapadma` | 537 | 272 | 50.7% | +87 | **359** | **66.9%** | **+16.2%** |
| `24-Bhramara` | 455 | 373 | 82.0% | +32 | **405** | **89.0%** | **+7.0%** |
| `25-Hamsassya` | 565 | 415 | 73.5% | +37 | **452** | **80.0%** | **+6.5%** |
| `26-Hamsapaksha` | 569 | 176 | 30.9% | +87 | **263** | **46.2%** | **+15.3%** |
| `28-Mukula` | 571 | 303 | 53.1% | +65 | **368** | **64.4%** | **+11.4%** |
| `30-Trishula` | 560 | 451 | 80.5% | +41 | **492** | **87.9%** | **+7.3%** |