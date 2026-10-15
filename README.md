\# Real-Time Bharatanatyam Mudra Recognition



A computer vision project that recognizes 30 Bharatanatyam single-hand mudras in real time using MediaPipe hand landmarks and a Support Vector Machine (SVM) classifier.



\## Features



\- Real-time video input using DroidCam and OpenCV

\- MediaPipe Hand Landmarker for 21 hand landmarks

\- 63 normalized landmark coordinates as model features

\- RBF-kernel SVM for 30-class classification

\- Prediction confidence and temporal smoothing

\- Confusion matrices, classification reports, and model comparisons



\## System Architecture



Phone Camera → DroidCam → OpenCV → MediaPipe → 21 Hand Landmarks → 63 Normalized Features → SVM → Mudra Prediction



\## Model Evaluation



Results on the held-out offline test dataset:



| Metric | Result |

|---|---:|

| Conditional test accuracy | 91.48% |

| Macro F1-score | 90.93% |

| Weighted F1-score | 91.46% |

| End-to-end coverage-aware accuracy | 83.90% |



Conditional metrics cover the 963 test images where hand landmarks were detected. End-to-end accuracy includes all 1,050 test images, counting hand-detection failures as unsuccessful outcomes.



These offline metrics do not guarantee equivalent performance with live camera input.



\## Project Structure



\- `src/` — auditing, feature extraction, model training, evaluation, and webcam recognition

\- `data/processed/` — extracted training and testing landmark features

\- `models/` — trained classifiers, model metadata, and the MediaPipe hand-landmarker model

\- `reports/` — evaluation reports, plots, and confusion matrices

\- `requirements.txt` — Python dependencies



The original raw dataset is not included in this repository.



\## Installation



Python 3.11 is recommended for the current development environment.



Create and activate a virtual environment:



```bash

python -m venv .venv

source .venv/Scripts/activate

python -m pip install --upgrade pip

pip install -r requirements.txt

```



\## Running the Application



1\. Open DroidCam on your phone.

2\. Connect through the DroidCam Windows client.

3\. Confirm the client displays the live phone-camera feed.

4\. Activate the Python virtual environment.

5\. Run:



```bash

python src/run\_webcam.py

```



Place your hand clearly in the camera view to see the predicted mudra and confidence. Press `q` or `Esc` to exit the recognition window.



\## Technologies Used



Python, OpenCV, MediaPipe, NumPy, pandas, scikit-learn, and joblib.



\## Limitations



Performance depends on lighting, hand visibility, camera position, and the similarity between mudras. The system may classify ambiguous gestures incorrectly or fail when hand landmarks cannot be detected.



Confirm the dataset's licence and usage conditions before redistributing the original data.

