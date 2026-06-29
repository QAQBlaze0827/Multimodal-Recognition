# Models

Put trained or downloaded ONNX models here.

Expected runtime paths:

```text
models/mini_xception_int8.onnx
models/tiny_cnn_audio_int8.onnx
```

Current runtime behavior:

- If `models/mini_xception_int8.onnx` exists, video emotion uses ONNX Runtime.
- If it does not exist, video emotion falls back to a lightweight heuristic so the camera pipeline can still be tested.
- If `models/tiny_cnn_audio_int8.onnx` exists, audio emotion uses ONNX Runtime.
- If it does not exist, audio emotion uses a simple fallback.

Recommended first milestone:

1. Train or download the video model first.
2. Verify `python app.py --vision-only`.
3. Add audio emotion after the video pipeline is stable.
