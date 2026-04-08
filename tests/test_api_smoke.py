import base64
import io
import json
import shutil
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
from PIL import Image

import app as webcam_app


def make_test_image_data_url(size=(128, 128), color=(240, 240, 240)):
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def make_combined_video_artifact():
    action_scores = {"Sitting": 0.93, "Standing": 0.07}
    emotion_scores = {"happy": 0.87, "neutral": 0.13}
    model_artifacts = {
        "mode": "combined",
        "mode_label": webcam_app.DETECTION_MODES["combined"]["label"],
        "required_models": webcam_app.DETECTION_MODES["combined"]["models"],
        "spatial_context": {
            "model_width": 128,
            "model_height": 128,
            "display_width": 128,
            "display_height": 128,
            "scale_x": 1.0,
            "scale_y": 1.0,
        },
        "objects": [
            {
                "id": "obj_1",
                "label": "chair",
                "confidence": 0.82,
                "bbox": {"x": 4, "y": 6, "width": 40, "height": 40},
                "coordinates": {"x1": 4, "y1": 6, "x2": 44, "y2": 46},
                "source_model": "object",
            }
        ],
        "humans": [
            {
                "id": "human_1",
                "label": "person",
                "confidence": 0.91,
                "bbox": {"x": 20, "y": 10, "width": 76, "height": 114},
                "coordinates": {"x1": 20, "y1": 10, "x2": 96, "y2": 124},
                "source_model": "human",
                "keypoints": [],
                "visible_keypoint_count": None,
                "keypoint_count": None,
                "summary": {"face": "Yes", "emotion": "Happy", "action": "Sitting"},
                "action": {"label": "Sitting", "confidence": 0.93, "scores": action_scores},
            }
        ],
        "faces": [
            {
                "id": "face_1",
                "label": "face",
                "confidence": 1.0,
                "bbox": {"x": 30, "y": 20, "width": 30, "height": 36},
                "coordinates": {"x1": 30, "y1": 20, "x2": 60, "y2": 56},
                "parent_human_id": "human_1",
                "source_model": "face",
                "landmarks": {
                    "left_eye": [36, 30],
                    "right_eye": [52, 30],
                    "nose": [44, 38],
                    "mouth_left": [38, 48],
                    "mouth_right": [50, 48],
                    "source": "cascade_heuristic",
                },
                "emotion": {"label": "happy", "confidence": 0.87, "scores": emotion_scores},
            }
        ],
        "emotions": [
            {
                "face_id": "face_1",
                "parent_human_id": "human_1",
                "label": "happy",
                "confidence": 0.87,
                "scores": emotion_scores,
            }
        ],
        "actions": [
            {
                "human_id": "human_1",
                "label": "Sitting",
                "confidence": 0.93,
                "scores": action_scores,
            }
        ],
    }
    phase1_output = {
        "frame_id": 1,
        "timestamp": "2026-04-08T12:00:00Z",
        "objects": [],
        "humans": [],
        "faces": [],
        "emotions": [],
        "actions": [],
        "notes": [],
        "detections": [],
        "model_artifacts": model_artifacts,
    }
    second_phase1_output = dict(phase1_output)
    second_phase1_output["frame_id"] = 2
    second_phase1_output["timestamp"] = "2026-04-08T12:00:01Z"
    second_phase1_output["model_artifacts"] = json.loads(json.dumps(model_artifacts))
    return {
        "selected_mode": "combined",
        "selected_mode_label": webcam_app.DETECTION_MODES["combined"]["label"],
        "model_manifest": {
            "selected_mode": "combined",
            "selected_mode_label": webcam_app.DETECTION_MODES["combined"]["label"],
            "required_models": webcam_app.DETECTION_MODES["combined"]["models"],
            "models": [],
        },
        "phase1_output": phase1_output,
        "inference_frames": [
            {"frame_index": 1, "video_time_seconds": 0.0, "detection_count": 3, "phase1_output": phase1_output},
            {"frame_index": 2, "video_time_seconds": 0.5, "detection_count": 3, "phase1_output": second_phase1_output},
        ],
    }


class DetectApiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        webcam_app.app.testing = True
        cls.client = webcam_app.app.test_client()

    def test_detect_json_face_mode_returns_success(self):
        if "face" not in webcam_app.AVAILABLE_MODE_KEYS:
            self.skipTest("Face mode is not available in this environment.")

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

    def test_face_emotion_pipeline_requires_human_then_face_before_emotion(self):
        frame = np.zeros((128, 128, 3), dtype=np.uint8)
        events = []
        human_detections = [
            {
                "x1": 16,
                "y1": 12,
                "x2": 96,
                "y2": 120,
                "confidence": 0.91,
                "label": "person",
                "source_model": "human",
            }
        ]

        def detect_objects_side_effect(_frame, mode="combined"):
            events.append(f"detect_objects:{mode}")
            return [dict(det) for det in human_detections]

        def detect_faces_side_effect(_frame):
            events.append("detect_faces")
            return []

        with (
            mock.patch.object(webcam_app, "detect_objects", side_effect=detect_objects_side_effect),
            mock.patch.object(webcam_app, "detect_faces", side_effect=detect_faces_side_effect),
            mock.patch.object(webcam_app, "detect_emotions") as detect_emotions,
        ):
            payload = webcam_app.run_phase1_pipeline(
                "face_emotion",
                frame,
                display_width=128,
                display_height=128,
            )

        self.assertEqual(events, ["detect_objects:human", "detect_faces"])
        detect_emotions.assert_not_called()
        self.assertEqual(len(payload["humans"]), 1)
        self.assertEqual(payload["faces"], [])
        self.assertEqual(payload["emotions"], [])

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

    def test_run_phase1_pipeline_can_include_model_artifacts(self):
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
        face_detections = [
            {
                "x1": 30,
                "y1": 20,
                "x2": 60,
                "y2": 56,
                "confidence": 1.0,
                "label": "face",
                "source_model": "face",
                "landmarks": {
                    "left_eye": [36, 30],
                    "right_eye": [52, 30],
                    "nose": [44, 38],
                    "mouth_left": [38, 48],
                    "mouth_right": [50, 48],
                    "source": "cascade_heuristic",
                },
            }
        ]

        def annotate_action(_image_np, detections):
            enriched = [dict(det) for det in detections]
            enriched[0]["action_label"] = "Sitting"
            enriched[0]["action_confidence"] = 0.93
            enriched[0]["action_scores"] = {"Sitting": 0.93, "Standing": 0.07}
            return enriched

        def detect_emotions(_image_np, detections):
            enriched = [dict(det) for det in detections]
            enriched[0]["emotion_label"] = "happy"
            enriched[0]["emotion_confidence"] = 0.87
            enriched[0]["emotion_scores"] = {"happy": 0.87, "neutral": 0.13}
            return enriched, [
                {
                    "face_id": "face_1",
                    "parent_human_id": "human_1",
                    "label": "happy",
                    "confidence": 0.87,
                }
            ]

        with (
            mock.patch.object(webcam_app, "run_combined_detections", return_value=(object_detections, human_detections)),
            mock.patch.object(webcam_app, "detect_faces", return_value=face_detections),
            mock.patch.object(webcam_app, "detect_emotions", side_effect=detect_emotions),
            mock.patch.object(webcam_app, "annotate_human_actions", side_effect=annotate_action),
        ):
            payload = webcam_app.run_phase1_pipeline(
                "combined",
                frame,
                display_width=128,
                display_height=128,
                include_model_artifacts=True,
            )

        artifacts = payload["model_artifacts"]
        self.assertEqual(artifacts["mode"], "combined")
        self.assertEqual(artifacts["required_models"], webcam_app.DETECTION_MODES["combined"]["models"])
        self.assertEqual(artifacts["objects"][0]["source_model"], "object")
        self.assertEqual(artifacts["humans"][0]["action"]["scores"]["Sitting"], 0.93)
        self.assertEqual(artifacts["faces"][0]["emotion"]["scores"]["happy"], 0.87)
        self.assertEqual(artifacts["faces"][0]["landmarks"]["nose"], [44, 38])
        self.assertEqual(artifacts["emotions"][0]["scores"]["happy"], 0.87)
        self.assertEqual(artifacts["actions"][0]["scores"]["Sitting"], 0.93)

    def test_build_filtered_video_job_payload_filters_saved_artifact_by_mode(self):
        artifact = make_combined_video_artifact()
        job_payload = webcam_app.build_video_job_context("job-1", "combined")

        compatible_modes = {mode["key"] for mode in webcam_app.filterable_detection_modes_for_artifact(artifact)}
        self.assertIn("face", compatible_modes)
        self.assertIn("action", compatible_modes)
        self.assertNotIn("pose", compatible_modes)

        filtered_payload = webcam_app.build_filtered_video_job_payload(job_payload, artifact, "face")

        self.assertTrue(filtered_payload["artifact_filter_applied"])
        self.assertEqual(filtered_payload["artifact_source_mode"], "combined")
        self.assertEqual(filtered_payload["selected_mode"], "face")
        self.assertEqual(filtered_payload["detection_count"], 2)
        self.assertEqual(len(filtered_payload["detections"]), 1)
        self.assertEqual(filtered_payload["detections"][0]["label"], "face")
        self.assertEqual(filtered_payload["phase1_output"]["humans"][0]["id"], "human_1")
        self.assertEqual(filtered_payload["phase1_output"]["faces"][0]["id"], "face_1")
        self.assertEqual(filtered_payload["phase1_output"]["emotions"], [])
        self.assertEqual(filtered_payload["phase1_output"]["actions"], [])

    def test_combined_mode_declares_action_dependency(self):
        self.assertIn("action", webcam_app.DETECTION_MODES["combined"]["models"])

    def test_face_modes_declare_human_dependency(self):
        self.assertIn("human", webcam_app.DETECTION_MODES["face"]["models"])
        self.assertIn("human", webcam_app.DETECTION_MODES["face_emotion"]["models"])

    def test_video_stride_defaults_allow_longer_processing(self):
        self.assertEqual(webcam_app.resolve_video_inference_stride(360, selected_mode="object"), 4)
        self.assertEqual(webcam_app.resolve_video_inference_stride(180, selected_mode="combined"), 6)

    def test_start_video_job_saves_upload_to_video_upload_dir(self):
        upload_dir = Path("test_video_upload_tmp")
        shutil.rmtree(upload_dir, ignore_errors=True)
        upload_dir.mkdir()

        class FakeFileStorage:
            filename = "clip.mp4"

            def __init__(self):
                self.saved_path = None

            def save(self, path):
                self.saved_path = Path(path)
                self.saved_path.write_bytes(b"video")

        class FakeThread:
            def __init__(self, target=None, args=None, daemon=None):
                self.target = target
                self.args = args
                self.daemon = daemon
                self.started = False

            def start(self):
                self.started = True

        fake_storage = FakeFileStorage()
        created_threads = []
        fake_uuids = [
            mock.Mock(hex="media123"),
            mock.Mock(hex="media456"),
            mock.Mock(hex="job456"),
        ]

        def build_thread(*args, **kwargs):
            thread = FakeThread(*args, **kwargs)
            created_threads.append(thread)
            return thread

        try:
            with (
                mock.patch.object(webcam_app, "VIDEO_UPLOAD_DIR", upload_dir),
                mock.patch.object(webcam_app, "uuid4", side_effect=fake_uuids),
                mock.patch.object(webcam_app.threading, "Thread", side_effect=build_thread),
            ):
                job_id = webcam_app.start_video_job(fake_storage, "combined")

            self.assertEqual(job_id, "job456")
            self.assertEqual(fake_storage.saved_path, upload_dir / "media_media123.mp4")
            self.assertTrue(fake_storage.saved_path.exists())
            self.assertEqual(webcam_app.VIDEO_JOBS[job_id]["selected_mode"], "combined")
            self.assertEqual(webcam_app.VIDEO_JOBS[job_id]["source_video_filename"], "media_media123.mp4")
            self.assertEqual(webcam_app.VIDEO_JOBS[job_id]["source_video_original_name"], "clip.mp4")
            self.assertEqual(webcam_app.VIDEO_JOBS[job_id]["source_video_path"], str(upload_dir / "media_media123.mp4"))
            self.assertEqual(len(created_threads), 1)
            self.assertTrue(created_threads[0].started)
            self.assertEqual(created_threads[0].target, webcam_app.process_video_job)
            self.assertEqual(
                created_threads[0].args,
                ("job456", upload_dir / "media_media123.mp4", "media_media456", "combined"),
            )
        finally:
            webcam_app.VIDEO_JOBS.pop("job456", None)
            shutil.rmtree(upload_dir, ignore_errors=True)

    def test_start_video_job_from_existing_reuses_saved_source_video(self):
        source_path = Path("test_reuse_video_tmp.mp4")
        source_path.write_bytes(b"video")
        webcam_app.VIDEO_JOBS["source-job"] = webcam_app.build_video_job_context(
            "source-job",
            "combined",
            media_type="video",
            job_status="completed",
            source_video_filename=source_path.name,
            source_video_original_name="original.mp4",
            source_video_path=str(source_path),
        )

        try:
            with mock.patch.object(webcam_app, "queue_video_job", return_value="new-job") as queue_job:
                job_id = webcam_app.start_video_job_from_existing("source-job", "face")

            self.assertEqual(job_id, "new-job")
            queue_job.assert_called_once_with(
                source_path,
                "face",
                source_video_original_name="original.mp4",
                source_video_origin_job_id="source-job",
            )
        finally:
            webcam_app.VIDEO_JOBS.pop("source-job", None)
            source_path.unlink(missing_ok=True)

    def test_detect_reuses_existing_video_when_source_job_id_is_posted(self):
        with mock.patch.object(webcam_app, "start_video_job_from_existing", return_value="job789") as start_job:
            response = self.client.post(
                "/detect",
                data={
                    "detection_mode": "face",
                    "source_video_job_id": "source-job",
                },
            )

        self.assertEqual(response.status_code, 302)
        self.assertIn("/image-detection?mode=face&job_id=job789", response.headers["Location"])
        start_job.assert_called_once_with("source-job", "face")

    def test_image_detection_filters_completed_video_artifact_without_reprocessing(self):
        job_id = "job-filter"
        webcam_app.VIDEO_JOBS[job_id] = webcam_app.build_video_job_context(
            job_id,
            "combined",
            media_type="video",
            job_status="completed",
            result_video_url="/generated-media/example.webm",
            result_json_url="/generated-media/example.json",
        )
        artifact = make_combined_video_artifact()

        try:
            with (
                mock.patch.object(webcam_app, "load_video_artifact", return_value=artifact),
                mock.patch.object(
                    webcam_app,
                    "render_image_detection_page",
                    side_effect=lambda selected_mode=None, **extra: webcam_app.jsonify(
                        {"selected_mode": selected_mode, **extra}
                    ),
                ),
            ):
                response = self.client.get(f"/image-detection?job_id={job_id}&mode=face")

            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertEqual(payload["selected_mode"], "face")
            self.assertTrue(payload["artifact_filter_applied"])
            self.assertEqual(payload["artifact_source_mode"], "combined")
            self.assertEqual(payload["detection_count"], 2)
            self.assertEqual(payload["detections"][0]["label"], "face")
            available_mode_keys = [mode["key"] for mode in payload["available_detection_modes"]]
            self.assertIn("combined", available_mode_keys)
            self.assertIn("face", available_mode_keys)
            self.assertIn("action", available_mode_keys)
            self.assertIn("object", available_mode_keys)
            self.assertNotIn("pose", available_mode_keys)
        finally:
            webcam_app.VIDEO_JOBS.pop(job_id, None)

    def test_process_video_job_writes_json_artifact(self):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        phase1_output = {
            "frame_id": 1,
            "timestamp": "2026-04-08T12:00:00Z",
            "objects": [],
            "humans": [],
            "faces": [],
            "emotions": [],
            "actions": [],
            "notes": [],
            "detections": [
                {
                    "id": "obj_1",
                    "label": "chair",
                    "confidence": 0.88,
                    "x1": 4,
                    "y1": 6,
                    "x2": 18,
                    "y2": 20,
                    "overlay_lines": ["Chair (0.88)"],
                    "display_label": "Chair (0.88)",
                }
            ],
        }

        class FakeCapture:
            def __init__(self, frame_bgr):
                self.frame_bgr = frame_bgr
                self.read_count = 0

            def isOpened(self):
                return True

            def get(self, prop):
                if prop == webcam_app.cv2.CAP_PROP_FPS:
                    return 24.0
                if prop == webcam_app.cv2.CAP_PROP_FRAME_COUNT:
                    return 1
                if prop == webcam_app.cv2.CAP_PROP_FRAME_WIDTH:
                    return self.frame_bgr.shape[1]
                if prop == webcam_app.cv2.CAP_PROP_FRAME_HEIGHT:
                    return self.frame_bgr.shape[0]
                return 0

            def read(self):
                if self.read_count == 0:
                    self.read_count += 1
                    return True, self.frame_bgr.copy()
                return False, None

            def release(self):
                return None

        class FakeWriter:
            def write(self, _frame_bgr):
                return None

            def release(self):
                return None

        generated_media_dir = Path("test_generated_media_tmp")
        shutil.rmtree(generated_media_dir, ignore_errors=True)
        generated_media_dir.mkdir()
        job_id = "job-123"
        try:
            input_path = generated_media_dir / "upload.mp4"
            input_path.write_bytes(b"input")
            output_path = generated_media_dir / "media_test_annotated.mp4"
            output_path.write_bytes(b"video")
            webcam_app.VIDEO_JOBS[job_id] = webcam_app.build_video_job_context(job_id, "combined")

            with (
                mock.patch.object(webcam_app, "GENERATED_MEDIA_DIR", generated_media_dir),
                mock.patch.object(webcam_app.cv2, "VideoCapture", return_value=FakeCapture(frame)),
                mock.patch.object(
                    webcam_app,
                    "create_compatible_video_writer",
                    return_value=(FakeWriter(), output_path, "video/mp4"),
                ),
                mock.patch.object(webcam_app, "run_phase1_pipeline", return_value=phase1_output),
                mock.patch.object(webcam_app, "annotate_display_detections", side_effect=lambda image_bgr, _detections: image_bgr),
            ):
                webcam_app.process_video_job(job_id, input_path, "media_test", "combined")

            artifact_path = generated_media_dir / "media_test.json"
            self.assertTrue(artifact_path.exists())
            artifact_payload = json.loads(artifact_path.read_text(encoding="utf-8"))

            self.assertEqual(artifact_payload["job_status"], "completed")
            self.assertEqual(artifact_payload["selected_mode"], "combined")
            self.assertEqual(artifact_payload["result_video_url"], "/generated-media/media_test_annotated.mp4")
            self.assertEqual(artifact_payload["result_json_url"], "/generated-media/media_test.json")
            self.assertEqual(artifact_payload["artifact_version"], 2)
            self.assertEqual(artifact_payload["selected_mode_label"], webcam_app.DETECTION_MODES["combined"]["label"])
            self.assertEqual(artifact_payload["detections"], phase1_output["detections"])
            self.assertEqual(artifact_payload["phase1_output"], phase1_output)
            self.assertEqual(artifact_payload["video_processing"]["inference_frame_count"], 1)
            self.assertEqual(artifact_payload["video_processing"]["preview_frame_index"], 1)
            self.assertEqual(artifact_payload["inference_frames"][0]["frame_index"], 1)
            self.assertEqual(artifact_payload["inference_frames"][0]["phase1_output"], phase1_output)
            self.assertEqual(artifact_payload["model_manifest"]["selected_mode"], "combined")
            self.assertTrue(
                any(model_entry["required_for_selected_mode"] for model_entry in artifact_payload["model_manifest"]["models"])
            )
            self.assertNotIn("source_video_path", artifact_payload)
            self.assertTrue(input_path.exists())
            self.assertEqual(
                webcam_app.VIDEO_JOBS[job_id]["result_json_url"],
                "/generated-media/media_test.json",
            )
        finally:
            webcam_app.VIDEO_JOBS.pop(job_id, None)
            shutil.rmtree(generated_media_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
