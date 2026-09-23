"""Nhan dien cu chi tay: ghep HandTracker (dinh vi 21 landmark, xem
hand_tracker.py) + KeypointClassifier (phan loai hinh dang ban tay tu
landmark, model train boi train_keypoint_classifier.py tu du lieu thu boi
collect_landmark_data.py).

Truoc day file nay tu phan loai xoe/nam bang nguong goc gap PIP + ty le xoe
ngon cai do thu cong (xem lich su git) - phai doan/hieu chinh tung nguong
(FINGER_STRAIGHT_ANGLE_DEG, THUMB_SPREAD_RATIO...) rieng cho tung nguoi
dung/anh sang/goc camera, va them 1 cu chi moi nghia la phai nghi ra 1 bo
dieu kien hinh hoc moi tu dau. Thay bang mot classifier NHE (MLP, xem
train_keypoint_classifier.py) tu HOC duong bien phan loai tu du lieu
landmark that - them cu chi moi chi can thu them du lieu (collect_landmark_
data.py) roi train lai, khong can sua logic.

hand_landmarker.task (dinh vi 21 khop tay) VAN la model co san cua Google -
buoc nay khong tranh duoc vi day la uoc luong toa do khop, khong phai buoc
"phan loai cu chi" (buoc da bi thay the tu dau du an la GestureRecognizer
canned cua Google - bo tin cay tut khi xoay tay, xem hand_tracker.py).
Rieng buoc PHAN LOAI cu chi tu 21 toa do do la model tu train, khong dung
ban co san.
"""

import config
from hand_tracker import HandTracker
from keypoint_classifier import KeypointClassifier
from landmark_utils import landmarks_to_feature_vector

# Phai khop CHINH XAC voi noi dung model/keypoint_classifier_label.csv
# (dong 0 = class id 0, dong 1 = class id 1) - xem collect_landmark_data.py.
OPEN_PALM = "OPEN_PALM"
FIST = "FIST"


class HandGestureDetector:
    def __init__(self):
        self._tracker = HandTracker()
        self._classifier = KeypointClassifier(config.KEYPOINT_MODEL_PATH, config.KEYPOINT_LABEL_PATH)

    def process(self, frame_rgb, image_width, image_height):
        """Tra ve (gesture, confidence, hand_landmarks).

        hand_landmarks la None neu khong thay tay. gesture la None neu
        khong thay tay HOAC do tin cay du doan duoi
        config.CLASSIFIER_CONFIDENCE_THRESHOLD (chua ro cu chi - tuong tu
        vung "dead zone" giua 2 nguong goc cua cach lam cu). confidence la
        0.0 khi khong thay tay.

        image_width/image_height PHAI la kich thuoc frame GOC (chua flip)
        dung de detect - phai giong het kich thuoc dung khi thu du lieu
        train (xem landmark_utils.landmarks_to_feature_vector)."""
        hand_landmarks = self._tracker.detect(frame_rgb)
        if hand_landmarks is None:
            return None, 0.0, None

        feature_vector = landmarks_to_feature_vector(hand_landmarks, image_width, image_height)
        label, confidence = self._classifier.predict(feature_vector)
        gesture = label if confidence >= config.CLASSIFIER_CONFIDENCE_THRESHOLD else None
        return gesture, confidence, hand_landmarks

    def close(self):
        self._tracker.close()
