#!/usr/bin/env python3
"""
src/run_webcam.py - Real-Time Bharatanatyam Mudra Recognition using DroidCam / Webcam.

Pipeline:
1. Automatic Camera Detection (probes camera indexes 0 to 5, prints available cameras & resolutions).
2. MediaPipe HandLandmarker Tasks API (models/hand_landmarker.task) in IMAGE mode.
3. 63-feature landmark extraction and scale/wrist normalization (src/features.py).
4. Pre-trained SVM model inference (models/final_svm_model.joblib) with probability confidence.
5. Temporal smoothing (rolling window majority voting) to prevent frame flicker.
6. Real-time annotated HUD overlay with 21-point hand skeleton and mudra classification.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
import logging
from pathlib import Path
import sys
import time
from typing import Any, Deque, List, Optional, Tuple

import cv2
import joblib
import numpy as np

# Add src to sys.path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from audit_dataset import MediaPipeHandDetector
from features import (
    draw_hand_landmarks_on_image,
    extract_features_from_mediapipe_result,
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("WebcamRecognition")

# ==============================================================================
# CAMERA CONFIGURATION
# ==============================================================================
# Default to camera index 0 with cv2.CAP_MSMF backend for DroidCam / Webcam.
CAMERA_INDEX: int = 0
CAMERA_BACKEND: int = cv2.CAP_MSMF


def detect_available_cameras(max_tested: int = 6) -> List[Tuple[int, int, int]]:
    """Probes OpenCV camera indexes 0 to max_tested - 1 with CAP_MSMF.
    
    Returns:
        List of tuples: (camera_index, width, height) for each working camera.
    """
    working_cameras: List[Tuple[int, int, int]] = []

    for idx in range(max_tested):
        cap = cv2.VideoCapture(idx, CAMERA_BACKEND)
        if not cap.isOpened():
            cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                working_cameras.append((idx, w, h))
            cap.release()

    return working_cameras


def select_camera(camera_override: Optional[int] = None) -> int:
    """Finds and selects the appropriate camera index.
    
    Follows DroidCam configuration rules:
    - If camera_override or CAMERA_INDEX is set, uses that directly.
    - Otherwise tests indexes 0 through 5 and auto-selects the working camera.
    """
    print("Starting Bharatanatyam Mudra Recognition...")

    target_index = camera_override if camera_override is not None else CAMERA_INDEX

    if target_index is not None:
        print(f"Selected camera index: {target_index} (Backend: CAP_MSMF)\n")
        return target_index

    print("Searching for cameras...")
    working_cameras = detect_available_cameras(max_tested=6)

    if not working_cameras:
        print("\n" + "=" * 60)
        print("ERROR: OpenCV could not access any working camera.")
        print("Please check that:")
        print("  1. DroidCam is running on your phone and Windows laptop.")
        print("  2. In the DroidCam PC client, click 'Start' to begin streaming.")
        print("  3. Your phone and laptop are on the same Wi-Fi or connected via USB.")
        print("=" * 60 + "\n")
        sys.exit(1)

    print("\nAvailable cameras:")
    for idx, w, h in working_cameras:
        print(f"Camera {idx} (Resolution: {w}x{h})")

    # In Windows DroidCam setups, Camera 0 or 1 is the virtual DroidCam device
    chosen_idx = working_cameras[0][0]
    print(f"\nSelected camera: {chosen_idx}\n")

    return chosen_idx


class MudraTemporalSmoother:
    """Applies temporal smoothing across recent prediction frames using majority voting."""

    def __init__(self, window_size: int = 7) -> None:
        self.window_size = window_size
        self.history: Deque[Tuple[str, float]] = deque(maxlen=window_size)

    def update(self, prediction: str, confidence: float) -> Tuple[str, float]:
        self.history.append((prediction, confidence))

        classes = [p for p, _ in self.history]
        counts = Counter(classes)
        majority_class, count = counts.most_common(1)[0]

        # Average confidence for the majority class
        confidences = [c for p, c in self.history if p == majority_class]
        avg_conf = float(np.mean(confidences)) if confidences else confidence

        return majority_class, avg_conf

    def clear(self) -> None:
        self.history.clear()


def draw_hud(
    frame: np.ndarray,
    hand_detected: bool,
    mudra_name: str,
    confidence: float,
    fps: float,
    camera_idx: int,
) -> np.ndarray:
    """Draws an informative, high-contrast HUD banner on the video frame."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Semi-transparent top HUD bar
    overlay = out.copy()
    hud_h = 105
    cv2.rectangle(overlay, (0, 0), (w, hud_h), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    # Accent color line
    accent_color = (0, 220, 100) if hand_detected else (0, 140, 255)
    cv2.line(out, (0, hud_h), (w, hud_h), accent_color, 2)

    # Title & camera index
    cv2.putText(
        out,
        "Bharatanatyam Mudra Recognition (DroidCam / Webcam)",
        (15, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    cam_info = f"Cam: {camera_idx} | FPS: {fps:.1f}"
    cv2.putText(
        out,
        cam_info,
        (w - 180, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    if hand_detected and mudra_name:
        # Display detected mudra
        mudra_text = f"Mudra: {mudra_name}"
        cv2.putText(
            out,
            mudra_text,
            (15, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        conf_text = f"Confidence: {confidence * 100:.1f}%"
        cv2.putText(
            out,
            conf_text,
            (15, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 230, 120),
            1,
            cv2.LINE_AA,
        )

        # Draw mini confidence bar
        bar_x = 210
        bar_y = 78
        bar_w = 150
        bar_h = 14
        fill_w = int(bar_w * np.clip(confidence, 0.0, 1.0))
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), 1)
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 220, 100), -1)

    else:
        # Waiting for hand message
        cv2.putText(
            out,
            "No Hand Detected - Show Hand to Camera",
            (15, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            "Position your right/left hand clearly in the frame",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

    # Footer key shortcut hint
    cv2.putText(
        out,
        "Press 'q' or ESC to exit",
        (15, h - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    return out


def reopen_camera(cam_idx: int, backend: int) -> Optional[cv2.VideoCapture]:
    """Attempts a clean release-and-reopen of the camera using the specified backend."""
    logger.info(f"Attempting clean release-and-reopen of camera {cam_idx}...")
    try:
        new_cap = cv2.VideoCapture(cam_idx, backend)
        if not new_cap.isOpened():
            new_cap = cv2.VideoCapture(cam_idx)
        if new_cap.isOpened():
            time.sleep(1.0)
            ret, _ = new_cap.read()
            if ret:
                logger.info("Camera re-opened successfully and supplying frames.")
                return new_cap
            new_cap.release()
    except Exception as ex:
        logger.error(f"Error during camera reopen: {ex}")
    return None


def run_webcam(
    camera_index: Optional[int] = None,
    model_path: Path = Path("models/final_svm_model.joblib"),
    landmark_model_path: Path = Path("models/hand_landmarker.task"),
    max_frames: Optional[int] = None,
    save_sample_path: Optional[Path] = None,
) -> None:
    """Runs the live webcam recognition loop."""
    # 1. Camera Detection & Selection
    selected_camera = select_camera(camera_override=camera_index)

    # 2. Load trained SVM pipeline
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found at: {model_path}")
    logger.info(f"Loading trained SVM classifier pipeline from: {model_path}")
    svm_model = joblib.load(model_path)

    # 3. Open Video Capture using cv2.CAP_MSMF
    logger.info(f"Connecting to camera {selected_camera} using cv2.CAP_MSMF...")
    cap = cv2.VideoCapture(selected_camera, CAMERA_BACKEND)
    if not cap.isOpened():
        logger.warning(f"Could not open camera {selected_camera} with CAP_MSMF. Falling back to default backend...")
        cap = cv2.VideoCapture(selected_camera)
        if not cap.isOpened():
            logger.error(f"Failed to open camera index {selected_camera}.")
            sys.exit(1)

    # Camera warm-up: wait 2.0 seconds and discard 30 initial frames paced at ~0.03s
    logger.info("Warming up camera for 2.0 seconds...")
    time.sleep(2.0)
    logger.info("Reading and discarding 30 initial warm-up frames (paced at ~0.03s)...")
    warmup_success = 0
    for i in range(30):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            warmup_success += 1
        else:
            logger.warning(f"Warm-up frame {i+1}/30 read failed (ret={ret})")
        time.sleep(0.03)
    logger.info(f"Camera warm-up complete: {warmup_success}/30 frames read successfully.")

    # Test five consecutive camera reads after warm-up
    logger.info("Testing 5 consecutive post-warmup camera reads...")
    post_warmup_ok = 0
    for i in range(5):
        ret, test_frame = cap.read()
        if ret and test_frame is not None and test_frame.size > 0:
            post_warmup_ok += 1
            logger.info(f"  Post-warmup test read {i+1}/5: SUCCESS (shape={test_frame.shape})")
        else:
            logger.warning(f"  Post-warmup test read {i+1}/5: FAILED (ret={ret})")
        time.sleep(0.03)
    logger.info(f"Post-warmup test result: {post_warmup_ok}/5 succeeded.")

    if post_warmup_ok == 0:
        logger.warning("Post-warmup reads failed. Attempting one clean camera reopen...")
        cap.release()
        new_cap = reopen_camera(selected_camera, CAMERA_BACKEND)
        if new_cap is None:
            logger.error("Camera failed post-warmup test and reopen failed. Exiting.")
            sys.exit(1)
        cap = new_cap

    smoother = MudraTemporalSmoother(window_size=7)

    # FPS Calculation
    prev_time = time.time()
    fps = 0.0

    frames_processed = 0
    hands_detected_count = 0
    predictions_made: List[str] = []

    print("=" * 60)
    print("Mudra recognition active! Press 'q' or ESC on the video window to quit.")
    print("=" * 60)

    debug_frame_saved = False
    debug_frame_path = Path("reports/webcam_debug_frame.jpg")

    consecutive_failures = 0
    reopen_attempted = False
    max_consecutive_failures = 10

    try:
        # 4. Initialize MediaPipe HandLandmarker in IMAGE mode
        with MediaPipeHandDetector(model_path=landmark_model_path, num_hands=1) as detector:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    if consecutive_failures <= 3:
                        logger.warning(f"Camera frame read failed (consecutive failure {consecutive_failures}).")
                    elif consecutive_failures == 4:
                        logger.warning("Multiple camera read failures. Suppressing continuous log messages...")

                    if consecutive_failures >= max_consecutive_failures:
                        if not reopen_attempted:
                            logger.warning(
                                f"Reached {max_consecutive_failures} consecutive failures. "
                                "Attempting one clean release-and-reopen..."
                            )
                            reopen_attempted = True
                            cap.release()
                            new_cap = reopen_camera(selected_camera, CAMERA_BACKEND)
                            if new_cap is not None:
                                cap = new_cap
                                consecutive_failures = 0
                                time.sleep(0.05)
                                continue
                            else:
                                logger.error("Camera reopen failed after read failures. Stopping recognition loop.")
                                break
                        else:
                            logger.error("Camera stopped supplying frames after reopen. Stopping recognition loop.")
                            break

                    time.sleep(0.03)
                    continue

                consecutive_failures = 0
                frames_processed += 1

                # Save one raw frame from running loop for verification
                if not debug_frame_saved:
                    debug_frame_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(debug_frame_path), frame)
                    debug_frame_saved = True
                    logger.info(
                        f"Saved running loop debug frame to: {debug_frame_path} "
                        f"(shape: {frame.shape}, mean: {frame.mean():.2f})"
                    )

                # Calculate live FPS
                curr_time = time.time()
                fps = 1.0 / max(curr_time - prev_time, 1e-5)
                prev_time = curr_time

                # 5. Run MediaPipe Hand Landmark Detection
                # Original BGR frame is preserved; MediaPipe receives frame (which detector converts to RGB internally)
                detected, result, det_err = detector.detect(frame)

                mudra_name = ""
                confidence = 0.0

                # 6. Draw landmarks and interface onto a copy of the actual camera frame
                display_frame = frame.copy()

                if detected and result and result.hand_landmarks:
                    hands_detected_count += 1
                    # Extract 63-feature normalized landmark vector
                    features = extract_features_from_mediapipe_result(result)

                    if features is not None and features.shape == (63,):
                        # SVM prediction & probability confidence
                        raw_pred = svm_model.predict(features.reshape(1, -1))[0]
                        probs = svm_model.predict_proba(features.reshape(1, -1))[0]
                        raw_conf = float(np.max(probs))

                        # Apply temporal smoothing
                        mudra_name, confidence = smoother.update(raw_pred, raw_conf)
                        predictions_made.append(mudra_name)

                    # Overlay 21-point hand skeleton onto display_frame
                    first_hand = result.hand_landmarks[0]
                    display_frame = draw_hand_landmarks_on_image(
                        display_frame,
                        first_hand,
                        draw_connections=True,
                        draw_points=True,
                        connection_color=(255, 215, 0),
                        point_color=(0, 90, 255),
                        point_radius=4,
                        line_thickness=2,
                    )
                else:
                    smoother.clear()

                # Render HUD on display_frame copy
                annotated_frame = draw_hud(
                    frame=display_frame,
                    hand_detected=detected,
                    mudra_name=mudra_name,
                    confidence=confidence,
                    fps=fps,
                    camera_idx=selected_camera,
                )

                # Save sample frame if requested
                if save_sample_path and frames_processed == 1:
                    save_sample_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_sample_path), annotated_frame)
                    logger.info(f"Saved initial webcam sample to: {save_sample_path}")

                # Display frame
                try:
                    cv2.imshow("Bharatanatyam Mudra Recognition", annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                        logger.info("Quit key pressed. Exiting...")
                        break
                except Exception:
                    # In non-GUI environments, allow headless processing
                    pass

                # Exit if max_frames reached (for testing/automation)
                if max_frames is not None and frames_processed >= max_frames:
                    logger.info(f"Reached max frames limit ({max_frames}). Stopping test run.")
                    if save_sample_path:
                        cv2.imwrite(str(save_sample_path), annotated_frame)
                    break

    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    # Print run summary
    print("\n" + "=" * 50)
    print("WEBCAM SESSION SUMMARY")
    print("=" * 50)
    print(f"Selected camera index:      {selected_camera}")
    print(f"Frames successfully read:   {frames_processed}")
    print(f"Hand detections count:      {hands_detected_count}")
    if predictions_made:
        most_common_mudra, count = Counter(predictions_made).most_common(1)[0]
        print(f"Mudra predictions count:    {len(predictions_made)}")
        print(f"Dominant prediction:        {most_common_mudra} ({count} frames)")
    else:
        print("Mudra predictions:          None (no hand detected in camera view)")
    print("=" * 50 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-Time Bharatanatyam Mudra Recognition using DroidCam / Webcam",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Camera index override (e.g. 0, 1). If omitted, uses auto-detection.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/final_svm_model.joblib"),
        help="Path to final trained SVM model",
    )
    parser.add_argument(
        "--landmark-model",
        type=Path,
        default=Path("models/hand_landmarker.task"),
        help="Path to MediaPipe hand_landmarker.task model",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process before exiting (useful for automated testing)",
    )
    parser.add_argument(
        "--save-sample",
        type=Path,
        default=Path("reports/webcam_sample.jpg"),
        help="Path to save a sample captured/annotated frame",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_webcam(
            camera_index=args.camera,
            model_path=args.model_path,
            landmark_model_path=args.landmark_model,
            max_frames=args.max_frames,
            save_sample_path=args.save_sample,
        )
    except Exception as e:
        logger.exception(f"Webcam recognition failed: {e}")
        sys.exit(1)
