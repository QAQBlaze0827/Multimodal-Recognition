from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG: dict[str, Any] = {
    "system": {"mode": "local", "log_level": "INFO"},
    "video": {
        "camera_id": 0,
        "width": 320,
        "height": 240,
        "fps_target": 15,
        "frame_skip": 2,
        "face_detection": {
            "backend": "auto",
            "min_detection_confidence": 0.7,
            "model_selection": 0,
        },
        "model": {
            "path": "models/mini_xception_int8.onnx",
            "input_size": 48,
            "input_channels": 1,
        },
    },
    "audio": {
        "enabled": True,
        "sample_rate": 16000,
        "chunk_size": 256,
        "device_id": -1,
        "window_seconds": 1.5,
        "mfcc": {"n_mfcc": 13, "n_fft": 512, "hop_length": 256},
        "model": {"path": "models/tiny_cnn_audio_int8.onnx"},
    },
    "fusion": {
        "method": "confidence_weighted",
        "fallback_emotion": "neutral",
        "min_confidence": 0.3,
    },
    "output": {
        "display": {
            "show_video": True,
            "show_audio": True,
            "show_fused": True,
            "show_fps": True,
            "show_temp": True,
        },
        "csv_log": {"enabled": True, "path": "data/logs/session_{timestamp}.csv"},
    },
    "thermal": {
        "check_interval": 5,
        "thresholds": {"cautious": 65, "eco": 75, "critical": 85},
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path = "config/config.yaml") -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists() or yaml is None:
        return deepcopy(DEFAULT_CONFIG)
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return deep_merge(DEFAULT_CONFIG, loaded)
