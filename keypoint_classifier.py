"""Wrapper suy luan cho model phan loai cu chi tu feature vector landmark
(xem landmark_utils.py). Model duoc TRAIN RIENG boi
train_keypoint_classifier.py tu du lieu thu boi collect_landmark_data.py -
file nay chi nap model da train va du doan, khong train.
"""

import csv

import joblib


class KeypointClassifier:
    def __init__(self, model_path, label_path):
        try:
            self._model = joblib.load(model_path)
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Khong tim thay model '{model_path}'. Can train truoc: "
                "1) python collect_landmark_data.py  (thu du lieu landmark that), "
                "2) python train_keypoint_classifier.py  (train + luu model), "
                "roi chay lai chuong trinh nay."
            ) from e

        with open(label_path, encoding="utf-8-sig") as f:
            self._labels = [row[0] for row in csv.reader(f) if row]

    def predict(self, feature_vector):
        """Tra ve (nhan_du_doan, do_tin_cay [0,1]) - do_tin_cay la xac suat
        (predict_proba) cua lop co xac suat cao nhat, dung de so sanh voi
        config.CLASSIFIER_CONFIDENCE_THRESHOLD o gesture_detector.py."""
        probabilities = self._model.predict_proba([feature_vector])[0]
        best_idx = int(probabilities.argmax())
        return self._labels[best_idx], float(probabilities[best_idx])
