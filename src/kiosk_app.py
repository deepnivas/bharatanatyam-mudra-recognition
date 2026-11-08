#!/usr/bin/env python3
"""
src/kiosk_app.py - Interactive Kuchipudi Mudra Recognition and Correction Kiosk (Part 8).

Features:
- Dual-pane widescreen touchscreen-friendly layout:
    * LEFT: Live webcam feed with MediaPipe 21-point hand tracking & color-coded joint feedback
    * RIGHT: Instructor-approved reference pose image & reference skeleton overlay
    * BOTTOM: Status banner, mudra comparison, and finger-specific correction instructions
- 4-State visual feedback:
    * STATE A: CORRECT (Vibrant Green)
    * STATE B: WRONG MUDRA (Vibrant Red)
    * STATE C: POSITION NEEDS CORRECTION (Amber / Orange)
    * STATE D: UNABLE TO ASSESS (Subtle Blue / Gray)
- Touchscreen-friendly interactive buttons:
    * [ Next Mudra ]
    * [ Previous ]
    * [ Try Again ]
    * [ Repeat Voice ]
    * [ Model Switcher (V1-V5) ]
- Non-blocking audio feedback (synthesized chimes + optional Windows SAPI TTS voice)
- Lesson progress tracking indicator (e.g. Mudra 1 of 30)
- Automated testing / headless support via --max-frames
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from correction_engine import (
    FingerStatus,
    MudraCorrectionEngine,
    MudraCorrectionReport,
    MudraEvaluationState,
)
from features import HAND_CONNECTIONS, draw_hand_landmarks_on_image
from recognition_engine import (
    MudraRecognitionEngine,
    MudraRecognitionResult,
)

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("KioskApp")


# ==============================================================================
# AUDIO & VOICE FEEDBACK MANAGER
# ==============================================================================

class AudioFeedbackManager:
    """Non-blocking audio chime and speech feedback manager."""

    def __init__(self, sample_rate: int = 44100, enabled: bool = True) -> None:
        self.sample_rate = sample_rate
        self.enabled = enabled
        self.last_played_state: Optional[MudraEvaluationState] = None
        self.last_played_time: float = 0.0
        self.min_interval: float = 3.0  # seconds between repeated voice cues

    def play_tone(self, frequencies: List[float], durations: List[float]) -> None:
        """Plays synthesized pleasant audio tones non-blockingly."""
        if not self.enabled or not AUDIO_AVAILABLE:
            return

        def _worker():
            try:
                tones = []
                for freq, dur in zip(frequencies, durations):
                    t = np.linspace(0, dur, int(self.sample_rate * dur), endpoint=False)
                    # Envelope to avoid audio clicks
                    envelope = np.sin(np.pi * np.linspace(0, 1, len(t)))
                    wave = 0.3 * np.sin(2 * np.pi * freq * t) * envelope
                    tones.append(wave)
                audio_data = np.concatenate(tones).astype(np.float32)
                sd.play(audio_data, self.sample_rate)
                sd.wait()
            except Exception as e:
                logger.debug(f"Audio playback error: {e}")

        threading.Thread(target=_worker, daemon=True).start()

    def speak(self, text: str) -> None:
        """Speaks instruction text asynchronously without blocking the UI thread."""
        if not self.enabled:
            return

        def _speech_worker():
            try:
                # Use Windows SAPI via powershell / python
                import subprocess
                clean_text = text.replace('"', '').replace("'", "")
                cmd = f'PowerShell -Command "Add-Type –AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{clean_text}\');"'
                subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                logger.debug(f"Speech error: {e}")

        threading.Thread(target=_speech_worker, daemon=True).start()

    def notify_state(self, state: MudraEvaluationState, instruction: str = "") -> None:
        """Plays state-appropriate chime and spoken feedback."""
        now = time.time()
        if state == self.last_played_state and (now - self.last_played_time) < self.min_interval:
            return

        self.last_played_state = state
        self.last_played_time = now

        if state == MudraEvaluationState.STATE_A_CORRECT:
            # Ascending success chime: C5 (523Hz) -> E5 (659Hz) -> G5 (784Hz)
            self.play_tone([523.25, 659.25, 783.99], [0.10, 0.10, 0.25])
            self.speak("Correct mudra!")
        elif state == MudraEvaluationState.STATE_B_WRONG_MUDRA:
            # Descending alert chime: G4 (392Hz) -> D4 (293Hz)
            self.play_tone([392.00, 293.66], [0.12, 0.20])
            if instruction:
                self.speak(instruction)
        elif state == MudraEvaluationState.STATE_C_INCORRECT_EXECUTION:
            # Gentle double guidance chime: A4 (440Hz) -> A4 (440Hz)
            self.play_tone([440.00, 440.00], [0.08, 0.15])
            if instruction:
                self.speak(instruction)


# ==============================================================================
# UI BUTTON COMPONENT
# ==============================================================================

class KioskButton:
    """Touchscreen button with hover and active states."""

    def __init__(
        self,
        rect: Tuple[int, int, int, int],  # (x, y, w, h)
        label: str,
        callback: Callable[[], None],
        bg_color: Tuple[int, int, int] = (50, 50, 50),
        active_color: Tuple[int, int, int] = (0, 140, 255),
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        self.rect = rect
        self.label = label
        self.callback = callback
        self.bg_color = bg_color
        self.active_color = active_color
        self.text_color = text_color
        self.is_hovered = False

    def contains(self, px: int, py: int) -> bool:
        x, y, w, h = self.rect
        return x <= px <= x + w and y <= py <= y + h

    def draw(self, canvas: np.ndarray) -> None:
        x, y, w, h = self.rect
        color = self.active_color if self.is_hovered else self.bg_color

        # Draw rounded/bordered rectangle
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (180, 180, 180), 1)

        # Draw centered text
        font_scale = 0.52
        thickness = 1
        t_size, _ = cv2.getTextSize(self.label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        tx = x + max(4, (w - t_size[0]) // 2)
        ty = y + (h + t_size[1]) // 2 - 2

        cv2.putText(
            canvas,
            self.label,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            self.text_color,
            thickness,
            cv2.LINE_AA,
        )


# ==============================================================================
# MAIN KIOSK APPLICATION
# ==============================================================================

class InteractiveMudraKiosk:
    """Main interactive digital instructor kiosk application."""

    def __init__(
        self,
        camera_index: Optional[int] = 0,
        default_model: str = "V4",
        screen_size: Tuple[int, int] = (1280, 720),
        enable_audio: bool = True,
    ) -> None:
        self.screen_w, self.screen_h = screen_size
        self.camera_index = camera_index
        self.enable_audio = enable_audio

        # Core Engines
        logger.info("Initializing Recognition Engine...")
        self.rec_engine = MudraRecognitionEngine(default_model=default_model)

        logger.info("Initializing Correction Engine...")
        self.corr_engine = MudraCorrectionEngine()

        self.audio_manager = AudioFeedbackManager(enabled=enable_audio)

        # Lesson state
        self.supported_mudras = self.corr_engine.get_supported_mudras()
        if not self.supported_mudras:
            logger.warning("No mudras found in reference library! Building defaults...")
            self.supported_mudras = ["1-Pataka"]

        self.current_mudra_idx = 0
        self.current_model_key = default_model
        self.last_report: Optional[MudraCorrectionReport] = None

        # Reference cache
        self.cached_ref_img: Optional[np.ndarray] = None
        self.cached_ref_mudra: str = ""

        # UI Buttons setup
        self.buttons: List[KioskButton] = []
        self._init_buttons()

        self.window_name = "Kuchipudi Mudra Recognition & Correction Kiosk"

    def close(self) -> None:
        """Closes recognition engine resources cleanly."""
        if hasattr(self, "rec_engine") and self.rec_engine is not None:
            self.rec_engine.close()

    @property
    def current_expected_mudra(self) -> str:
        return self.supported_mudras[self.current_mudra_idx]

    def _init_buttons(self) -> None:
        """Positions touch buttons across the bottom action bar."""
        btn_y = self.screen_h - 48
        btn_h = 38

        self.btn_prev = KioskButton(
            rect=(15, btn_y, 110, btn_h),
            label="< Previous",
            callback=self.previous_mudra,
            bg_color=(45, 45, 45),
            active_color=(0, 140, 255),
        )

        self.btn_next = KioskButton(
            rect=(135, btn_y, 110, btn_h),
            label="Next Mudra >",
            callback=self.next_mudra,
            bg_color=(45, 45, 45),
            active_color=(0, 140, 255),
        )

        self.btn_try_again = KioskButton(
            rect=(255, btn_y, 110, btn_h),
            label="Try Again",
            callback=self.try_again,
            bg_color=(45, 45, 45),
            active_color=(0, 200, 100),
        )

        self.btn_repeat = KioskButton(
            rect=(375, btn_y, 130, btn_h),
            label="Repeat Voice",
            callback=self.repeat_instruction,
            bg_color=(45, 45, 45),
            active_color=(180, 100, 255),
        )

        self.btn_model = KioskButton(
            rect=(self.screen_w - 240, btn_y, 225, btn_h),
            label=f"Model: {self.current_model_key}",
            callback=self.cycle_model,
            bg_color=(60, 60, 60),
            active_color=(255, 120, 0),
        )

        self.buttons = [
            self.btn_prev,
            self.btn_next,
            self.btn_try_again,
            self.btn_repeat,
            self.btn_model,
        ]

    def next_mudra(self) -> None:
        self.current_mudra_idx = (self.current_mudra_idx + 1) % len(self.supported_mudras)
        self.try_again()
        logger.info(f"Switched to next mudra: {self.current_expected_mudra}")

    def previous_mudra(self) -> None:
        self.current_mudra_idx = (self.current_mudra_idx - 1) % len(self.supported_mudras)
        self.try_again()
        logger.info(f"Switched to previous mudra: {self.current_expected_mudra}")

    def try_again(self) -> None:
        self.rec_engine.clear_smoothing()
        self.audio_manager.last_played_state = None
        logger.info("Pose recognition reset. Ready for next attempt.")

    def repeat_instruction(self) -> None:
        if self.last_report and self.last_report.instructions:
            self.audio_manager.speak(self.last_report.instructions[0])

    def cycle_model(self) -> None:
        keys = list(self.rec_engine.AVAILABLE_MODELS.keys())
        idx = keys.index(self.current_model_key)
        new_key = keys[(idx + 1) % len(keys)]
        self.current_model_key = new_key
        self.rec_engine.set_active_model(new_key)
        self.btn_model.label = f"Model: {new_key}"
        self.try_again()
        logger.info(f"Model cycled to: {self.rec_engine.AVAILABLE_MODELS[new_key]['name']}")

    def handle_mouse_event(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Handles touchscreen taps and mouse clicks."""
        if event == cv2.EVENT_MOUSEMOVE:
            for btn in self.buttons:
                btn.is_hovered = btn.contains(x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            for btn in self.buttons:
                if btn.contains(x, y):
                    btn.callback()
                    break

    def get_reference_visual(self, expected_mudra: str, target_size: Tuple[int, int]) -> np.ndarray:
        """Loads and prepares instructor-approved reference visualization with skeleton overlay."""
        tw, th = target_size
        if self.cached_ref_mudra == expected_mudra and self.cached_ref_img is not None:
            return cv2.resize(self.cached_ref_img, (tw, th))

        ref_data = self.corr_engine.get_reference_data(expected_mudra)
        vis = np.full((th, tw, 3), 25, dtype=np.uint8)

        if ref_data:
            img_path = Path(ref_data["image_path"])
            if img_path.is_file():
                raw_img = cv2.imread(str(img_path))
                if raw_img is not None:
                    vis = cv2.resize(raw_img, (tw, th))

            # Overlay instructor reference skeleton in pristine green/cyan
            lms = ref_data["landmarks"]
            vis = draw_hand_landmarks_on_image(
                vis,
                lms,
                draw_connections=True,
                draw_points=True,
                connection_color=(0, 220, 100),   # Instructor reference green
                point_color=(0, 255, 255),        # Cyan joints
                point_radius=4,
                line_thickness=2,
            )

        # Header badge on reference window
        cv2.rectangle(vis, (0, 0), (tw, 36), (20, 20, 20), -1)
        cv2.putText(
            vis,
            "INSTRUCTOR-APPROVED REFERENCE POSE",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 230, 120),
            1,
            cv2.LINE_AA,
        )

        self.cached_ref_mudra = expected_mudra
        self.cached_ref_img = vis
        return vis

    def render_kiosk_frame(
        self,
        camera_frame: np.ndarray,
        rec_result: MudraRecognitionResult,
        report: MudraCorrectionReport,
        fps: float,
    ) -> np.ndarray:
        """Composes the complete widescreen dual-pane kiosk layout."""
        canvas = np.full((self.screen_h, self.screen_w, 3), 18, dtype=np.uint8)

        # 1. TOP HEADER BANNER
        header_h = 56
        cv2.rectangle(canvas, (0, 0), (self.screen_w, header_h), (28, 28, 28), -1)
        cv2.line(canvas, (0, header_h), (self.screen_w, header_h), (50, 50, 50), 2)

        # Title
        cv2.putText(
            canvas,
            "KUCHIPUDI MUDRA RECOGNITION & CORRECTION KIOSK",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Lesson Progress Badge
        progress_text = f"Mudra {self.current_mudra_idx + 1} of {len(self.supported_mudras)}"
        cv2.putText(
            canvas,
            progress_text,
            (self.screen_w - 420, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 100),
            1,
            cv2.LINE_AA,
        )

        # Camera & FPS Info
        info_text = f"Cam: {self.camera_index} | FPS: {fps:.1f}"
        cv2.putText(
            canvas,
            info_text,
            (self.screen_w - 180, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        # 2. DUAL-PANE VIEWPORT
        pane_y = header_h + 10
        pane_h = 470
        pane_w = (self.screen_w - 30) // 2  # ~625px each

        # Left Pane: User Live Feed & Skeleton
        left_x = 10
        annotated_cam = camera_frame
        if rec_result.hand_detected and rec_result.landmarks_21_3 is not None:
            # Draw color-coded correction skeleton on user hand
            annotated_cam = self.corr_engine.draw_correction_skeleton(
                camera_frame,
                rec_result.landmarks_21_3,
                report.finger_statuses,
            )

        cam_resized = cv2.resize(annotated_cam, (pane_w, pane_h))
        canvas[pane_y : pane_y + pane_h, left_x : left_x + pane_w] = cam_resized
        cv2.rectangle(canvas, (left_x, pane_y), (left_x + pane_w, pane_y + pane_h), (60, 60, 60), 1)

        # Left Pane Sub-header
        cv2.rectangle(canvas, (left_x, pane_y), (left_x + pane_w, pane_y + 30), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            "YOUR LIVE PERFORMANCE",
            (left_x + 10, pane_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

        # Right Pane: Instructor Reference Pose
        right_x = left_x + pane_w + 10
        ref_visual = self.get_reference_visual(self.current_expected_mudra, (pane_w, pane_h))
        canvas[pane_y : pane_y + pane_h, right_x : right_x + pane_w] = ref_visual
        cv2.rectangle(canvas, (right_x, pane_y), (right_x + pane_w, pane_y + pane_h), (60, 60, 60), 1)

        # 3. BOTTOM EVALUATION & FEEDBACK SECTION
        status_y = pane_y + pane_h + 10
        status_h = 115
        cv2.rectangle(canvas, (10, status_y), (self.screen_w - 10, status_y + status_h), (25, 25, 25), -1)

        # Color-coded state banner
        state_colors = {
            MudraEvaluationState.STATE_A_CORRECT: (0, 200, 80),            # Green
            MudraEvaluationState.STATE_B_WRONG_MUDRA: (0, 0, 230),          # Red
            MudraEvaluationState.STATE_C_INCORRECT_EXECUTION: (0, 160, 255),# Amber
            MudraEvaluationState.STATE_D_UNABLE_TO_ASSESS: (160, 160, 160), # Slate/Gray
        }
        accent = state_colors.get(report.state, (100, 100, 100))
        cv2.rectangle(canvas, (10, status_y), (18, status_y + status_h), accent, -1)

        # Mudra Comparison Labels
        exp_text = f"Expected: {self.current_expected_mudra}"
        cv2.putText(canvas, exp_text, (30, status_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        rec_color = (0, 220, 100) if rec_result.is_expected_match else (0, 140, 255)
        rec_text = f"Recognized: {rec_result.predicted_mudra} ({rec_result.confidence*100:.1f}%)" if rec_result.hand_detected else "Recognized: No Hand Detected"
        cv2.putText(canvas, rec_text, (30, status_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.60, rec_color, 1, cv2.LINE_AA)

        # Status Badge
        cv2.putText(canvas, f"STATUS: {report.state.value}", (420, status_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, accent, 2, cv2.LINE_AA)

        # Finger-specific status indicators (5 small chips)
        chip_x = 420
        chip_y = status_y + 42
        for f_name, f_stat in report.finger_statuses.items():
            col = (0, 200, 80) if f_stat == FingerStatus.CORRECT else ((0, 160, 255) if f_stat == FingerStatus.NEEDS_ADJUSTMENT else (0, 0, 220))
            cv2.rectangle(canvas, (chip_x, chip_y), (chip_x + 65, chip_y + 18), (40, 40, 40), -1)
            cv2.rectangle(canvas, (chip_x, chip_y), (chip_x + 65, chip_y + 18), col, 1)
            cv2.putText(canvas, f"{f_name[:3]}: {f_stat.value[0]}", (chip_x + 5, chip_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)
            chip_x += 72

        # Primary instruction message
        inst_text = report.instructions[0] if report.instructions else report.explanation
        cv2.putText(
            canvas,
            f"Instruction: {inst_text}",
            (30, status_y + 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

        # 4. DRAW TOUCHSCREEN BUTTONS
        for btn in self.buttons:
            btn.draw(canvas)

        return canvas

    def run(self, max_frames: Optional[int] = None, save_sample_path: Optional[Path] = None) -> None:
        """Executes the main interactive kiosk loop."""
        print("=" * 60)
        print("STARTING KUCHIPUDI MUDRA RECOGNITION & CORRECTION KIOSK")
        print("Press 'n' for Next Mudra | 'p' for Prev | 'm' to Cycle Model | 'q' to Quit")
        print("=" * 60)

        cap = cv2.VideoCapture(self.camera_index if self.camera_index is not None else 0)
        if not cap.isOpened():
            logger.warning(f"Could not open camera {self.camera_index}. Creating synthetic test feed...")
            cap = None

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.screen_w, self.screen_h)
        cv2.setMouseCallback(self.window_name, self.handle_mouse_event)

        frames_count = 0
        prev_time = time.time()
        fps = 0.0

        try:
            while True:
                if cap is not None and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        frame = np.full((480, 640, 3), 40, dtype=np.uint8)
                else:
                    # Synthetic clean preview frame
                    frame = np.full((480, 640, 3), 35, dtype=np.uint8)

                frames_count += 1
                curr_time = time.time()
                fps = 1.0 / max(1e-5, curr_time - prev_time)
                prev_time = curr_time

                # 1. Run Recognition
                rec_result, _ = self.rec_engine.process_frame(
                    frame,
                    expected_mudra=self.current_expected_mudra,
                )

                # 2. Run Correction & 4-State Analysis
                report = self.corr_engine.evaluate_pose(
                    rec_result,
                    expected_mudra=self.current_expected_mudra,
                )
                self.last_report = report

                # 3. Audio & Voice Feedback
                if rec_result.hand_detected:
                    lead_inst = report.instructions[0] if report.instructions else ""
                    self.audio_manager.notify_state(report.state, lead_inst)

                # 4. Render Kiosk Composition
                kiosk_canvas = self.render_kiosk_frame(frame, rec_result, report, fps)

                # Save sample if requested
                if save_sample_path and frames_count == 1:
                    save_sample_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_sample_path), kiosk_canvas)
                    logger.info(f"Saved initial kiosk UI snapshot: {save_sample_path}")

                # Display
                try:
                    cv2.imshow(self.window_name, kiosk_canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):  # 'q' or ESC
                        break
                    elif key in (ord("n"), ord("N")):
                        self.next_mudra()
                    elif key in (ord("p"), ord("P")):
                        self.previous_mudra()
                    elif key in (ord("t"), ord("T")):
                        self.try_again()
                    elif key in (ord("r"), ord("R")):
                        self.repeat_instruction()
                    elif key in (ord("m"), ord("M")):
                        self.cycle_model()
                except Exception:
                    pass

                if max_frames is not None and frames_count >= max_frames:
                    logger.info(f"Reached max frames test limit ({max_frames}). Stopping kiosk loop.")
                    if save_sample_path:
                        cv2.imwrite(str(save_sample_path), kiosk_canvas)
                    break

        finally:
            if cap is not None:
                cap.release()
            self.rec_engine.close()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        print("Kiosk session closed cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Interactive Kuchipudi Mudra Recognition and Correction Kiosk")
    parser.add_argument("--camera", type=int, default=0, help="Camera index override (e.g. 0, 1)")
    parser.add_argument("--model", type=str, default="V4", choices=["V1", "V2", "V3", "V4", "V5"], help="Initial model to use")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process (useful for automated testing)")
    parser.add_argument("--save-sample", type=Path, default=Path("reports/kiosk_ui_sample.jpg"), help="Path to save UI screenshot")
    args = parser.parse_args()

    kiosk = InteractiveMudraKiosk(camera_index=args.camera, default_model=args.model)
    kiosk.run(max_frames=args.max_frames, save_sample_path=args.save_sample)


if __name__ == "__main__":
    main()
