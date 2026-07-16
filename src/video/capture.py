from __future__ import annotations


class _Cv2Backend:
    def __init__(self, cv2, cap):
        self._cv2 = cv2
        self._cap = cap

    def read(self):
        return self._cap.read()

    def release(self):
        self._cap.release()


class _Picamera2Backend:
    def __init__(self, cv2, width, height):
        self._cv2 = cv2
        from picamera2 import Picamera2
        self._picam2 = Picamera2()
        config = self._picam2.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"},
        )
        self._picam2.configure(config)
        self._picam2.start()

    def read(self):
        try:
            frame = self._picam2.capture_array()
            if frame.shape[2] == 4:
                frame = self._cv2.cvtColor(frame, self._cv2.COLOR_RGBA2BGR)
            else:
                frame = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR)
            return True, frame
        except Exception:
            return False, None

    def release(self):
        try:
            self._picam2.stop()
            self._picam2.close()
        except Exception:
            pass


def open_camera(cv2, camera_id: int, width: int, height: int, fps: int):
    cap = cv2.VideoCapture(camera_id)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        ret, _ = cap.read()
        if ret:
            return _Cv2Backend(cv2, cap)
    cap.release()

    try:
        return _Picamera2Backend(cv2, width, height)
    except Exception as exc:
        raise RuntimeError(
            f"Could not open camera index {camera_id}. "
            f"OpenCV read failed, picamera2 fallback also failed: {exc}"
        )
