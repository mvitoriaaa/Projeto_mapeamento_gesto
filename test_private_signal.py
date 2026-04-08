import tempfile
import unittest
from pathlib import Path

import numpy as np

import main


def build_metrics(
    *,
    states,
    finger_count,
    ok_distance_ratio,
    extension_ratios,
    tip_above_wrist_ratios,
    tip_gap_ratios,
):
    return {
        "states": states,
        "finger_count": finger_count,
        "ok_distance_ratio": ok_distance_ratio,
        "extension_ratios": extension_ratios,
        "tip_above_wrist_ratios": tip_above_wrist_ratios,
        "tip_gap_ratios": tip_gap_ratios,
    }


def private_signal_metrics():
    return build_metrics(
        states={
            "thumb": False,
            "index": False,
            "middle": True,
            "ring": False,
            "pinky": False,
        },
        finger_count=1,
        ok_distance_ratio=0.82,
        extension_ratios={
            "thumb": 0.02,
            "index": 0.04,
            "middle": 0.48,
            "ring": 0.05,
            "pinky": 0.03,
        },
        tip_above_wrist_ratios={
            "thumb": 0.12,
            "index": 0.28,
            "middle": 0.72,
            "ring": 0.26,
            "pinky": 0.22,
        },
        tip_gap_ratios={
            "index_to_middle": 0.36,
            "ring_to_middle": 0.34,
            "pinky_to_middle": 0.40,
        },
    )


class PrivateSignalRuleTests(unittest.TestCase):
    def test_private_signal_is_detected_only_for_strict_signal_variant(self):
        prediction = main.detect_private_signal(private_signal_metrics())
        self.assertEqual(prediction.label, main.PRIVATE_SIGNAL_LABEL)
        self.assertGreaterEqual(prediction.confidence, 0.90)

    def test_common_public_gestures_do_not_trigger_private_signal(self):
        gestures = [
            build_metrics(
                states={
                    "thumb": False,
                    "index": True,
                    "middle": False,
                    "ring": False,
                    "pinky": False,
                },
                finger_count=1,
                ok_distance_ratio=0.75,
                extension_ratios={
                    "thumb": 0.01,
                    "index": 0.40,
                    "middle": 0.02,
                    "ring": 0.03,
                    "pinky": 0.02,
                },
                tip_above_wrist_ratios={
                    "thumb": 0.10,
                    "index": 0.68,
                    "middle": 0.22,
                    "ring": 0.20,
                    "pinky": 0.18,
                },
                tip_gap_ratios={
                    "index_to_middle": -0.12,
                    "ring_to_middle": 0.10,
                    "pinky_to_middle": 0.12,
                },
            ),
            build_metrics(
                states={
                    "thumb": False,
                    "index": True,
                    "middle": True,
                    "ring": False,
                    "pinky": False,
                },
                finger_count=2,
                ok_distance_ratio=0.86,
                extension_ratios={
                    "thumb": 0.01,
                    "index": 0.35,
                    "middle": 0.36,
                    "ring": 0.05,
                    "pinky": 0.04,
                },
                tip_above_wrist_ratios={
                    "thumb": 0.10,
                    "index": 0.64,
                    "middle": 0.66,
                    "ring": 0.24,
                    "pinky": 0.20,
                },
                tip_gap_ratios={
                    "index_to_middle": 0.02,
                    "ring_to_middle": 0.18,
                    "pinky_to_middle": 0.24,
                },
            ),
            build_metrics(
                states={
                    "thumb": True,
                    "index": True,
                    "middle": True,
                    "ring": True,
                    "pinky": True,
                },
                finger_count=5,
                ok_distance_ratio=0.95,
                extension_ratios={
                    "thumb": 0.30,
                    "index": 0.31,
                    "middle": 0.32,
                    "ring": 0.30,
                    "pinky": 0.27,
                },
                tip_above_wrist_ratios={
                    "thumb": 0.55,
                    "index": 0.62,
                    "middle": 0.66,
                    "ring": 0.60,
                    "pinky": 0.55,
                },
                tip_gap_ratios={
                    "index_to_middle": 0.05,
                    "ring_to_middle": -0.04,
                    "pinky_to_middle": -0.08,
                },
            ),
            build_metrics(
                states={
                    "thumb": False,
                    "index": False,
                    "middle": False,
                    "ring": False,
                    "pinky": False,
                },
                finger_count=0,
                ok_distance_ratio=0.55,
                extension_ratios={
                    "thumb": 0.02,
                    "index": 0.01,
                    "middle": 0.03,
                    "ring": 0.02,
                    "pinky": 0.01,
                },
                tip_above_wrist_ratios={
                    "thumb": 0.12,
                    "index": 0.15,
                    "middle": 0.18,
                    "ring": 0.14,
                    "pinky": 0.12,
                },
                tip_gap_ratios={
                    "index_to_middle": 0.01,
                    "ring_to_middle": 0.00,
                    "pinky_to_middle": -0.02,
                },
            ),
        ]

        for metrics in gestures:
            with self.subTest(metrics=metrics):
                prediction = main.detect_private_signal(metrics)
                self.assertEqual(prediction.label, "")

    def test_private_signal_requires_more_stable_frames_than_public_gestures(self):
        smoother = main.GestureSmoother()
        public_prediction = main.GesturePrediction("ok", 0.93, "rules")
        private_prediction = main.GesturePrediction(main.PRIVATE_SIGNAL_LABEL, 0.96, "rules-private")

        stable_public = None
        for _ in range(main.PUBLIC_STABILITY_MIN_COUNT):
            stable_public, _ = smoother.update(public_prediction)
        self.assertEqual(stable_public.label, "ok")

        smoother = main.GestureSmoother()
        stable_private = None
        for _ in range(main.PRIVATE_SIGNAL_MIN_COUNT - 1):
            stable_private, _ = smoother.update(private_prediction)
        self.assertEqual(stable_private.label, "")

        stable_private, _ = smoother.update(private_prediction)
        self.assertEqual(stable_private.label, main.PRIVATE_SIGNAL_LABEL)


class PrivateSignalIsolationTests(unittest.TestCase):
    def test_private_signal_is_sanitized_for_public_surfaces(self):
        prediction = main.GesturePrediction(main.PRIVATE_SIGNAL_LABEL, 0.97, "rules-private")
        public_prediction = main.sanitize_public_prediction(prediction)
        self.assertEqual(public_prediction.label, "")

    def test_private_signal_is_not_logged(self):
        original_logs_dir = main.LOGS_DIR

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                main.LOGS_DIR = Path(tmp_dir)
                logger = main.SessionLogger()
                logger.log(main.GesturePrediction(main.PRIVATE_SIGNAL_LABEL, 0.97, "rules-private"))
                lines = logger.path.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(lines), 1)
        finally:
            main.LOGS_DIR = original_logs_dir

    def test_private_signal_uses_secret_overlay_only(self):
        frame = np.zeros((120, 200, 3), dtype=np.uint8)
        prediction = main.GesturePrediction(main.PRIVATE_SIGNAL_LABEL, 0.98, "rules-private")
        main.draw_main_overlay(frame, prediction)
        self.assertGreater(int(frame.sum()), 0)


if __name__ == "__main__":
    unittest.main()
