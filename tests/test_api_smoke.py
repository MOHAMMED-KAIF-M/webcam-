import base64
import io
import unittest
from unittest import mock

import numpy as np
from PIL import Image

import app as webcam_app


def make_test_image_data_url(size=(128, 128), color=(240, 240, 240)):
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


class DetectApiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        webcam_app.app.testing = True
        cls.client = webcam_app.app.test_client()

    def test_detect_json_face_mode_returns_success(self):
        response = self.client.post(
            "/detect",
            json={
                "detection_mode": "face",
                "image": make_test_image_data_url(),
                "display_width": 128,
                "display_height": 128,
            },
        )

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["model_type"], "face")
        self.assertEqual(payload["detection_mode"], "face")
        self.assertEqual(payload["model_label"], "Face Detection")
        self.assertIn("detections", payload)
        self.assertIsInstance(payload["detections"], list)
        self.assertIn("faces", payload)
        self.assertIn("emotions", payload)

    def test_detect_json_without_image_returns_bad_request(self):
        response = self.client.post(
            "/detect",
            json={
                "detection_mode": "face",
            },
        )

        self.assertEqual(response.status_code, 400, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertEqual(payload, {"error": "No image provided"})

    def test_detect_json_pose_mode_returns_success_when_available(self):
        if "pose" not in webcam_app.AVAILABLE_MODE_KEYS:
            self.skipTest("Pose mode is not available in this environment.")

        response = self.client.post(
            "/detect",
            json={
                "detection_mode": "pose",
                "image": make_test_image_data_url(),
                "display_width": 128,
                "display_height": 128,
            },
        )

        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        payload = response.get_json()
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["model_type"], "pose")
        self.assertEqual(payload["detection_mode"], "pose")
        self.assertEqual(payload["model_label"], "Pose Detection (Keypoint R-CNN)")
        self.assertIn("humans", payload)
        self.assertIsInstance(payload["humans"], list)

    def test_action_mode_falls_back_to_pose_when_human_detector_misses(self):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        pose_detection = {
            "x1": 16,
            "y1": 12,
            "x2": 96,
            "y2": 120,
            "confidence": 0.91,
            "label": "person",
            "keypoints": [
                {"name": "nose", "x": 48, "y": 20, "score": 0.9, "visible": True},
                {"name": "left_shoulder", "x": 36, "y": 42, "score": 0.8, "visible": True},
            ],
            "visible_keypoint_count": 2,
            "keypoint_count": 2,
        }

        with (
            mock.patch.object(webcam_app, "run_detector_detection", return_value=[]),
            mock.patch.object(
                webcam_app,
                "model_is_available",
                side_effect=lambda model_type: model_type == "pose",
            ),
            mock.patch.object(webcam_app, "get_model_bundle", return_value={"runtime": "keypoint_rcnn"}),
            mock.patch.object(webcam_app, "run_pose_detection", return_value=[pose_detection]),
        ):
            detections = webcam_app.detect_objects(frame, mode="action")

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["label"], "person")
        self.assertEqual(detections[0]["source_model"], "pose")
        self.assertEqual(detections[0]["visible_keypoint_count"], 2)

    def test_combined_mode_runs_action_annotation_when_action_model_is_available(self):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        object_detections = [
            {
                "x1": 4,
                "y1": 6,
                "x2": 44,
                "y2": 46,
                "confidence": 0.82,
                "label": "chair",
                "source_model": "object",
            }
        ]
        human_detections = [
            {
                "x1": 20,
                "y1": 10,
                "x2": 96,
                "y2": 124,
                "confidence": 0.91,
                "label": "person",
                "source_model": "human",
            }
        ]

        def annotate_action(_image_np, detections):
            enriched = [dict(det) for det in detections]
            enriched[0]["action_label"] = "Sitting"
            enriched[0]["action_confidence"] = 0.93
            return enriched

        with (
            mock.patch.object(webcam_app, "run_combined_detections", return_value=(object_detections, human_detections)),
            mock.patch.object(webcam_app, "detect_faces", return_value=[]),
            mock.patch.object(webcam_app, "annotate_human_actions", side_effect=annotate_action),
        ):
            payload = webcam_app.run_phase1_pipeline("combined", frame, display_width=128, display_height=128)

        self.assertEqual(payload["actions"][0]["label"], "Sitting")
        self.assertEqual(payload["humans"][0]["summary"]["action"], "Sitting")
        self.assertTrue(
            any("Sitting" in " ".join(det.get("overlay_lines", [])) for det in payload["detections"])
        )
        self.assertNotIn("No action detected for the current human detections.", payload.get("notes", []))

    def test_combined_mode_declares_action_dependency(self):
        self.assertIn("action", webcam_app.DETECTION_MODES["combined"]["models"])


if __name__ == "__main__":
    unittest.main()
