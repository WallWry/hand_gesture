"""Thu thap du lieu landmark THAT de train keypoint classifier (xem
train_keypoint_classifier.py). Chay script nay TRUOC train_keypoint_
classifier.py, va truoc lan dau chay main.py (main.py can model da train).

Cach dung:
  python collect_landmark_data.py

  - Dua tay vao khung hinh, giu dung 1 cu chi (vd xoe ban tay).
  - Nhan phim SO (0, 1, ...) TUONG UNG voi cu chi dang gio - moi lan nhan
    ghi 1 dong (nhan, 42 gia tri feature) vao config.KEYPOINT_DATASET_PATH.
    Nhan phim NHIEU LAN, o NHIEU goc xoay/khoang cach/vi tri/anh sang khac
    nhau cho MOI cu chi (khuyen nghi it nhat 50-100 mau/cu chi) de model
    tong quat hoa tot, khong "hoc thuoc long" 1 tu the duy nhat.
  - So phim <-> ten cu chi duoc dinh nghia trong config.KEYPOINT_LABEL_PATH
    (dong 0 = phim '0', dong 1 = phim '1', v.v - mac dinh: 0=OPEN_PALM,
    1=FIST). Muon them cu chi moi: them 1 dong vao file label do, roi thu
    du lieu cho phim so tuong ung voi dong moi.
  - Nhan 'q' de thoat.

Du lieu duoc GHI THEM (append) vao file csv co san - chay lai script nhieu
lan (vd bo sung du lieu sau khi thay do tin cay model chua du) se KHONG
mat du lieu da thu truoc do.
"""

import csv
import os

import cv2

import config
from hand_tracker import HandTracker
from landmark_utils import landmarks_to_feature_vector


def _load_labels():
    with open(config.KEYPOINT_LABEL_PATH, encoding="utf-8-sig") as f:
        return [row[0] for row in csv.reader(f) if row]


def _append_sample(class_id, feature_vector):
    os.makedirs(os.path.dirname(config.KEYPOINT_DATASET_PATH), exist_ok=True)
    with open(config.KEYPOINT_DATASET_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([class_id, *feature_vector])


def main():
    labels = _load_labels()
    print("Cu chi duoc gan phim so:")
    for i, name in enumerate(labels):
        print(f"  [{i}] {name}")
    print("Nhan phim so tuong ung de ghi 1 mau (chi khi da thay tay), 'q' de thoat.\n")

    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Khong mo duoc webcam (kiem tra quyen truy cap camera / index 0).")

    tracker = HandTracker()
    sample_counts = [0] * len(labels)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_landmarks = tracker.detect(frame_rgb)

            display_frame = cv2.flip(frame, 1) if config.MIRROR_DISPLAY else frame
            status = "Da thay tay" if hand_landmarks is not None else "Chua thay tay"
            counts_text = " ".join(f"{name}:{c}" for name, c in zip(labels, sample_counts))
            cv2.putText(display_frame, status, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, counts_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("Thu thap du lieu cu chi (phim so = ghi mau, q = thoat)", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if hand_landmarks is not None and ord('0') <= key <= ord('9'):
                class_id = key - ord('0')
                if class_id < len(labels):
                    # Dung frame.shape (kich thuoc frame GOC, chua flip) -
                    # phai giong het cach main.py/gesture_detector.py goi
                    # landmarks_to_feature_vector luc suy luan.
                    feature_vector = landmarks_to_feature_vector(hand_landmarks, frame.shape[1], frame.shape[0])
                    _append_sample(class_id, feature_vector)
                    sample_counts[class_id] += 1
                    print(f"Da ghi mau cho '{labels[class_id]}' (tong: {sample_counts[class_id]})")
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
