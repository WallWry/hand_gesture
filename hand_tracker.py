"""Doc 21 diem landmark ban tay tu MediaPipe Tasks HandLandmarker (chi dinh
vi khop tay, KHONG phan loai cu chi - xem gesture_detector.py cho buoc do).

Tach rieng khoi gesture_detector.py vi ca main.py (qua
gesture_detector.HandGestureDetector) lan collect_landmark_data.py deu can
dinh vi landmark, tranh lap code khoi tao HandLandmarker + quan ly
timestamp o 2 noi.
"""

import mediapipe as mp

import config


class HandTracker:
    def __init__(self):
        vision = mp.tasks.vision
        options = vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=config.HAND_MODEL_PATH),
            # VIDEO (khong phai IMAGE) - dung cho luong frame webcam lien
            # tuc: MediaPipe dung ket qua cac frame truoc de tracking, giup
            # landmark on dinh/it rung hon giua cac frame thay vi detect lai
            # tu dau moi frame nhu mot anh doc lap khong lien quan.
            running_mode=vision.RunningMode.VIDEO,
            num_hands=config.NUM_HANDS,
            min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        # RunningMode.VIDEO doi hoi timestamp (ms) tang dan tuyet doi giua
        # cac lan goi. Dung bo dem frame thay vi dong ho thuc te - dam bao
        # tang dan chac chan, khong phu thuoc do phan giai/do tre cua clock.
        self._frame_idx = 0

    def detect(self, frame_rgb):
        """Tra ve list 21 NormalizedLandmark cua tay dau tien, hoac None neu
        khong thay tay (happy case: 1 tay, xem config.NUM_HANDS)."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._frame_idx += 1
        result = self._landmarker.detect_for_video(mp_image, self._frame_idx)

        if not result.hand_landmarks:
            return None
        return result.hand_landmarks[0]

    def close(self):
        self._landmarker.close()
