"""Train keypoint classifier tu du lieu landmark thu boi
collect_landmark_data.py. Chay SAU khi da thu du lieu, TRUOC khi chay
main.py lan dau (hoac sau khi thu them du lieu de cai thien model).

Chay: python train_keypoint_classifier.py

Dau vao: config.KEYPOINT_DATASET_PATH (csv khong header: cot 0 = class id,
42 cot con lai = feature vector, xem landmark_utils.landmarks_to_feature_
vector). Dau ra: config.KEYPOINT_MODEL_PATH (KeypointClassifier trong
keypoint_classifier.py nap file nay de suy luan).

Model la RandomForestClassifier cua scikit-learn - phu hop voi dataset nho
(vai tram mau thu bang tay) hon MLPClassifier: khong can tinh chinh so lan
lap/early-stopping (voi dataset nho, MLPClassifier + early_stopping=True se
tu trich ~10% train set lam validation noi bo - qua it mau de danh gia dang
tin cay, de dung training qua som, model chua kip hoc gi ma van "coi nhu
hoi tu" - day la nguyen nhan chinh khien 1 lan chay truoc chi dat ~56%
accuracy du bai toan xoe/nam tay de phan biet). RandomForest on dinh hon
nhieu trong dieu kien it du lieu vi hoc tung cay quyet dinh doc lap tren
tap con du lieu/dac trung roi lay da so, khong phu thuoc mot lan hoi tu
gradient duy nhat.
"""

import csv
import os

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import config


def _load_dataset():
    rows = []
    with open(config.KEYPOINT_DATASET_PATH, encoding="utf-8") as f:
        for row in csv.reader(f):
            if row:
                rows.append([float(v) for v in row])
    if not rows:
        raise SystemExit(
            f"'{config.KEYPOINT_DATASET_PATH}' rong. Chay 'python collect_landmark_data.py' "
            "de thu du lieu truoc."
        )
    data = np.array(rows)
    return data[:, 1:], data[:, 0].astype(int)


def _load_labels():
    with open(config.KEYPOINT_LABEL_PATH, encoding="utf-8-sig") as f:
        return [row[0] for row in csv.reader(f) if row]


def main():
    try:
        X, y = _load_dataset()
    except FileNotFoundError:
        raise SystemExit(
            f"Khong tim thay '{config.KEYPOINT_DATASET_PATH}'. Chay "
            "'python collect_landmark_data.py' de thu du lieu truoc."
        )
    labels = _load_labels()

    class_counts = np.bincount(y, minlength=len(labels))
    print(f"Tong so mau: {len(X)}")
    for class_id, count in enumerate(class_counts):
        name = labels[class_id] if class_id < len(labels) else f"?{class_id}"
        print(f"  [{class_id}] {name}: {count} mau")

    present_counts = class_counts[class_counts > 0]
    if len(present_counts) < 2:
        raise SystemExit("Can du lieu cho it nhat 2 cu chi de train. Chay collect_landmark_data.py them.")
    if present_counts.min() < 2:
        raise SystemExit(
            "Moi cu chi can it nhat 2 mau de chia train/test. Chay collect_landmark_data.py de thu them du lieu."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,  # gioi han do sau - tranh 1 cay hoc thuoc long tung mau khi dataset nho
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    present_class_ids = sorted(set(y_train) | set(y_test))
    print("\n=== Danh gia tren tap test ===")
    print(classification_report(
        y_test, y_pred,
        labels=present_class_ids,
        target_names=[labels[i] for i in present_class_ids],
    ))
    print("Confusion matrix (thu tu hang/cot theo class id tang dan):")
    print(confusion_matrix(y_test, y_pred, labels=present_class_ids))

    os.makedirs(os.path.dirname(config.KEYPOINT_MODEL_PATH), exist_ok=True)
    joblib.dump(model, config.KEYPOINT_MODEL_PATH)
    print(f"\nDa luu model vao {config.KEYPOINT_MODEL_PATH}")


if __name__ == "__main__":
    main()
