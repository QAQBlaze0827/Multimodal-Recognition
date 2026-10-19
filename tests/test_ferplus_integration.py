import copy
from contextlib import ExitStack
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from scripts import compare_face_models as comparison
from scripts import preview_ferplus as preview
from src import main as app
from src.config import DEFAULT_CONFIG
from src.output.db_logger import DbLogger
from src.shared_types import EMOTIONS, FaceDetection, make_result
from src.video import ferplus
from src.video.inference import VideoEmotionPipeline


class FerPlusModelTests(unittest.TestCase):
    def setUp(self):
        self.session = Mock()
        self.session.get_inputs.return_value = [
            SimpleNamespace(name="input", type="tensor(float)", shape=[1, 1, 64, 64])]
        self.session.get_outputs.return_value = [
            SimpleNamespace(name="output", type="tensor(float)", shape=[1, 8])]
        self.session.run.return_value = [np.array([[0, 1, 2, 3, 5, 0, 0, 0]], np.float32)]
        self.crop = np.random.default_rng(42).integers(0, 256, (80, 100, 3), dtype=np.uint8)

    def create_model(self, *, exists=True, data=b"verified", loaded=True):
        with patch.object(Path, "is_file", return_value=exists), \
                patch.object(Path, "read_bytes", return_value=data), \
                patch.object(ferplus, "FER_SHA256", hashlib.sha256(b"verified").hexdigest()), \
                patch.object(ferplus, "create_ort_session",
                             return_value=(self.session, "input") if loaded else (None, None)):
            return ferplus.FerPlusEmotionModel()

    def test_live_and_offline_share_the_same_functions(self):
        self.assertIs(comparison.fer_tensor, ferplus.fer_tensor)
        self.assertIs(comparison.fer_scores, ferplus.fer_scores)
        self.assertIs(preview.fer_tensor, ferplus.fer_tensor)
        self.assertIs(preview.fer_scores, ferplus.fer_scores)

    def test_prediction_matches_preview_and_keeps_video_source(self):
        before = self.crop.copy()
        model = self.create_model()
        result = model.predict(self.crop)
        expected = comparison.fer_scores(self.session.run.return_value[0])
        np.testing.assert_allclose(list(result.scores.values()), list(expected["four_scores"].values()))
        np.testing.assert_array_equal(self.session.run.call_args.args[1]["input"], comparison.fer_tensor(self.crop))
        np.testing.assert_array_equal(before, self.crop)
        self.assertEqual(result.label, "anger")
        self.assertEqual(result.source, "video")
        self.assertEqual(model.backend, "onnx-ferplus")
        self.assertEqual(tuple(result.scores), EMOTIONS)

    def test_all_four_labels_use_correct_indices(self):
        model = self.create_model()
        for index, label in zip((0, 1, 3, 4), EMOTIONS):
            logits = np.zeros((1, 8), np.float32)
            logits[0, index] = 5
            self.session.run.return_value = [logits]
            self.assertEqual(model.predict(self.crop).label, label)

    def test_zero_retained_mass_has_zero_confidence(self):
        self.session.run.return_value = [np.array([[-10000, -10000, 10000, -10000, -10000, 0, 0, 0]])]
        self.assertEqual(self.create_model().predict(self.crop).confidence, 0)

    def test_invalid_output_fails_instead_of_guessing(self):
        model = self.create_model()
        for output in (np.zeros((1, 7)), np.full((1, 8), np.nan)):
            self.session.run.return_value = [output]
            with self.assertRaises(ValueError):
                model.predict(self.crop)

    def test_missing_model_does_not_open_onnx(self):
        with patch.object(Path, "is_file", return_value=False), \
                patch.object(ferplus, "create_ort_session") as opening, self.assertRaises(FileNotFoundError):
            ferplus.FerPlusEmotionModel()
        opening.assert_not_called()

    def test_invalid_hash_does_not_open_onnx(self):
        with patch.object(Path, "is_file", return_value=True), \
                patch.object(Path, "read_bytes", return_value=b"wrong"), \
                patch.object(ferplus, "create_ort_session") as opening, self.assertRaises(ValueError):
            ferplus.FerPlusEmotionModel()
        opening.assert_not_called()

    def test_load_failure_has_no_heuristic_fallback(self):
        with self.assertRaises(RuntimeError):
            self.create_model(loaded=False)

    def test_invalid_input_or_output_contract_rejected(self):
        for shape in ([1, 1, 48, 48], [1, 64, 64, 1], [2, 1, 64, 64]):
            self.session.get_inputs.return_value[0].shape = shape
            with self.assertRaises(ValueError):
                self.create_model()
        self.session.get_inputs.return_value[0].shape = [1, 1, 64, 64]
        for shape in ([1, 7], [8], [2, 8]):
            self.session.get_outputs.return_value[0].shape = shape
            with self.assertRaises(ValueError):
                self.create_model()


class FerPlusApplicationTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((100, 120, 3), 127, np.uint8)
        self.face = FaceDetection((10, 20, 50, 60), 0.9, "fake")
        self.anger = make_result("video", {"anger": 1})
        self.sad = make_result("video", {"sad": 1})

    def test_cli_defaults_to_legacy_and_accepts_explicit_selection(self):
        self.assertEqual(app.build_parser().parse_args([]).video_model, "legacy")
        self.assertEqual(app.build_parser().parse_args(["--video-model", "ferplus"]).video_model, "ferplus")
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            app.build_parser().parse_args(["--video-model", "unknown"])

    def test_audio_only_rejects_ferplus_before_running(self):
        with patch("sys.argv", ["app.py", "--audio-only", "--video-model", "ferplus"]), \
                patch("sys.stderr"), patch.object(app, "run_audio_only") as audio, self.assertRaises(SystemExit):
            app.main()
        audio.assert_not_called()

    def test_invalid_ferplus_stops_before_camera_audio_or_loggers(self):
        args = app.build_parser().parse_args(["--video-model", "ferplus"])
        with patch.object(ferplus, "FerPlusEmotionModel", side_effect=ValueError("bad model")), \
                patch.object(app, "open_camera") as camera, patch.object(app, "AudioEmotionThread") as audio, \
                patch.object(app, "CsvLogger") as csv, patch.object(app, "DbLogger") as db, \
                self.assertRaises(ValueError):
            app.run_video_loop(copy.deepcopy(DEFAULT_CONFIG), args)
        for resource in (camera, audio, csv, db):
            resource.assert_not_called()

    def test_no_face_clears_cache_and_reacquisition_predicts_immediately(self):
        detector = Mock()
        detector.detect.side_effect = [self.face, self.face, None, self.face]
        model = Mock(backend="onnx-ferplus")
        model.predict.side_effect = [self.anger, self.sad]
        pipeline = VideoEmotionPipeline(cv2, detector, model, frame_skip=2)
        pipeline.process(self.frame)
        self.assertEqual(pipeline.process(self.frame).emotion.label, "anger")
        missing = pipeline.process(self.frame)
        self.assertIsNone(missing.face)
        self.assertEqual(missing.emotion.confidence, 0)
        self.assertEqual(pipeline.cached.confidence, 0)
        self.assertEqual(pipeline.process(self.frame).emotion.label, "sad")
        self.assertEqual(model.predict.call_count, 2)

    def run_loop(self, choice, *, audio=False, inference_error=False):
        config = copy.deepcopy(DEFAULT_CONFIG)
        config["backend"]["enabled"] = True
        config["fusion"]["gate_audio_with_face"] = True
        before = copy.deepcopy(config)
        cli = ["--video-model", choice, "--no-display", "--max-frames", "4"]
        if not audio:
            cli.append("--vision-only")
        args = app.build_parser().parse_args(cli)
        cap = Mock()
        cap.read.return_value = (True, self.frame)
        detector = Mock(warning=None)
        detector.detect.side_effect = [self.face, self.face, None, self.face]
        model = Mock(backend="onnx-ferplus" if choice == "ferplus" else "onnx", model_path=Path("test.onnx"))
        model.predict.side_effect = ValueError("inference failed") if inference_error else [self.anger, self.sad]
        audio_thread = Mock()

        def start_audio(state, *_):
            audio_thread.start.side_effect = lambda: setattr(state, "audio", make_result("audio", {"happy": 1}))
            return audio_thread

        with ExitStack() as stack:
            fer = stack.enter_context(patch.object(ferplus, "FerPlusEmotionModel", return_value=model))
            old = stack.enter_context(patch.object(app, "VideoEmotionModel", return_value=model))
            stack.enter_context(patch.object(app, "open_camera", return_value=cap))
            stack.enter_context(patch.object(app, "FaceDetector", return_value=detector))
            csv = stack.enter_context(patch.object(app, "CsvLogger")).return_value
            db = stack.enter_context(patch.object(app, "DbLogger")).return_value
            stack.enter_context(patch.object(app, "AudioEmotionModel"))
            audio_factory = stack.enter_context(patch.object(app, "AudioEmotionThread", side_effect=start_audio))
            stack.enter_context(patch.object(app, "read_cpu_temp", return_value=None))
            display = stack.enter_context(patch.object(cv2, "imshow"))
            stack.enter_context(patch("builtins.print"))
            if inference_error:
                with self.assertRaises(ValueError):
                    app.run_video_loop(config, args)
            else:
                app.run_video_loop(config, args)
            if choice == "ferplus":
                fer.assert_called_once_with()
                old.assert_not_called()
            else:
                old.assert_called_once_with(cv2, config["video"]["model"]["path"], 48)
                fer.assert_not_called()
            if audio:
                audio_thread.start.assert_called_once()
                audio_thread.join.assert_called_once()
            else:
                audio_factory.assert_not_called()
            cap.release.assert_called_once()
            detector.close.assert_called_once()
            csv.close.assert_called_once()
            db.close.assert_called_once()
            display.assert_not_called()
            self.assertEqual(config, before)
            return db.write.call_args_list

    def test_ferplus_reaches_existing_logging_and_fusion(self):
        rows = self.run_loop("ferplus")
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[1].args[0].emotion.label, "anger")
        self.assertEqual(rows[1].args[2].label, "anger")
        self.assertEqual(rows[2].args[0].emotion.confidence, 0)
        self.assertEqual(rows[2].args[2].confidence, 0)
        self.assertEqual(rows[3].args[2].label, "sad")

    def test_legacy_still_uses_configured_model(self):
        self.assertEqual(len(self.run_loop("legacy")), 4)

    def test_audio_fusion_and_face_gate_are_preserved(self):
        rows = self.run_loop("ferplus", audio=True)
        self.assertEqual(rows[1].args[1].label, "happy")
        self.assertGreater(rows[1].args[2].scores["happy"], 0)
        self.assertGreater(rows[1].args[2].scores["anger"], 0)
        self.assertIsNone(rows[2].args[1])
        self.assertEqual(rows[2].args[2].confidence, 0)

    def test_inference_failure_releases_resources(self):
        self.run_loop("ferplus", inference_error=True)

    def test_existing_database_payload_keeps_frontend_contract(self):
        rows = self.run_loop("ferplus")
        logger = object.__new__(DbLogger)
        logger._enabled = True
        logger.log_interval = 0
        logger.last_log_time = 0
        logger.total_frames = 0
        logger.session_id = "mock-session"
        logger.db_path = "not-opened.db"
        with patch("src.output.db_logger.insert_log") as insert:
            for row in rows:
                logger.write(*row.args)
        payloads = [call.args[1] for call in insert.call_args_list]
        self.assertEqual(payloads[1]["video_emotion"], "anger")
        self.assertEqual(payloads[1]["video_conf"], 1)
        self.assertEqual(tuple(payloads[1]["video_scores"]), EMOTIONS)
        self.assertFalse(payloads[2]["face_detected"])
        self.assertEqual(payloads[2]["video_conf"], 0)


if __name__ == "__main__":
    unittest.main()
