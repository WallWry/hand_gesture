"""Thu thap du lieu landmark THAT de train keypoint classifier (xem
train_keypoint_classifier.py). Chay script nay TRUOC train_keypoint_
classifier.py, va truoc lan dau chay main.py (main.py can model da train).

Cach dung - GHI THEO DOT (burst capture):
  python collect_landmark_data.py

  - Dua tay vao khung hinh, giu dung 1 cu chi (vd xoe ban tay).
  - Nhan phim SO (0, 1, ...) TUONG UNG voi cu chi dang gio -> script tu
    dong GHI LIEN TUC trong config.BURST_CAPTURE_SECONDS giay (mac dinh 4s).
    TRONG LUC DO, GIU NGUYEN cu chi nhung CHU DONG DI CHUYEN TAY de 1 dot
    ghi tao ra nhieu mau DA DANG cung luc, thay vi phai bam phim tay hang
    tram lan cho tung tu the tinh:
      * Xoay co tay qua trai/phai (long ban tay huong dan thay vi luon
        chinh dien camera).
      * Nghieng tay len/xuong.
      * Dua tay GAN roi LUI XA camera.
      * Di chuyen tay NGANG (trai/phai/giua khung hinh) va DOC (cao/thap).
    Moi doi di chuyen o TREN chi la 1 dot rieng - nen ghi NHIEU DOT
    (khuyen nghi >= 5-6 dot/cu chi), MOI DOT chi tap trung 1-2 kieu di
    chuyen o tren cho de kiem soat, thay vi co lam tat ca cung luc.
  - Neu co the, doi them ANH SANG/NEN/VI TRI DUNG that su khac nhau giua
    cac dot (vd gan cua so ban ngay, den phong, gan tuong don sac...) -
    giup model chiu duoc dieu kien thuc te da dang hon la chi doi goc tay.
  - So phim <-> ten cu chi dinh nghia trong config.KEYPOINT_LABEL_PATH
    (dong 0 = phim '0', dong 1 = phim '1'... mac dinh: 0=OPEN_PALM,
    1=FIST). Them cu chi moi: them 1 dong vao file label do, roi ghi du
    lieu cho phim so tuong ung voi dong moi.
  - Nhan 'q' de thoat (ca giua 1 dot ghi).

Du lieu duoc GHI THEM (append) vao file csv co san - chay lai script
nhieu lan (vd bo sung du lieu sau khi thay do chinh xac model chua du) se
KHONG mat du lieu da thu truoc do.
"""

import csv
import os
import time

import cv2

import config
from hand_tracker import HandTracker
from landmark_utils import landmarks_to_feature_vector

_WINDOW_NAME = "Thu thap du lieu cu chi (phim so = ghi 1 dot, q = thoat)"


def _load_labels():
    with open(config.KEYPOINT_LABEL_PATH, encoding="utf-8-sig") as f:
        return [row[0] for row in csv.reader(f) if row]


def _append_sample(class_id, feature_vector):
    os.makedirs(os.path.dirname(config.KEYPOINT_DATASET_PATH), exist_ok=True)
    with open(config.KEYPOINT_DATASET_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([class_id, *feature_vector])


def _draw_idle_overlay(display_frame, hand_landmarks, labels, sample_counts):
    status = "Da thay tay" if hand_landmarks is not None else "Chua thay tay"
    counts_text = " ".join(f"{name}:{c}" for name, c in zip(labels, sample_counts))
    cv2.putText(display_frame, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(display_frame, counts_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(display_frame, "Bam phim so de ghi 1 dot ~4s - di chuyen tay trong luc ghi",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)


def _run_burst_capture(cap, tracker, class_id, class_name, sample_counts):
    """Ghi lien tuc trong config.BURST_CAPTURE_SECONDS giay, lay mau moi
    config.BURST_SAMPLE_EVERY_N_FRAMES frame (xem giai thich trong
    config.py). Tra ve True neu nguoi dung nhan 'q' de thoat hoan toan
    (ke ca giua dot ghi), False neu dot ghi ket thuc binh thuong."""
    deadline = time.time() + config.BURST_CAPTURE_SECONDS
    frame_i = 0
    captured = 0

    while time.time() < deadline:
        ok, frame = cap.read()
        if not ok:
            continue
        frame_i += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_landmarks = tracker.detect(frame_rgb)

        if hand_landmarks is not None and frame_i % config.BURST_SAMPLE_EVERY_N_FRAMES == 0:
            # Dung frame.shape (kich thuoc frame GOC, chua flip) - phai
            # giong het cach main.py/gesture_detector.py goi
            # landmarks_to_feature_vector luc suy luan.
            feature_vector = landmarks_to_feature_vector(hand_landmarks, frame.shape[1], frame.shape[0])
            _append_sample(class_id, feature_vector)
            captured += 1
            sample_counts[class_id] += 1

        display_frame = cv2.flip(frame, 1) if config.MIRROR_DISPLAY else frame
        remaining = max(0.0, deadline - time.time())
        cv2.putText(display_frame, f"DANG GHI '{class_name}': con {remaining:.1f}s - XOAY/NGHIENG/GAN-XA/DI CHUYEN TAY",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(display_frame, f"Da ghi dot nay: {captured} mau",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow(_WINDOW_NAME, display_frame)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            print(f"Da dung dot ghi '{class_name}' som (+{captured} mau, tong: {sample_counts[class_id]})")
            return True

    print(f"Ket thuc 1 dot ghi '{class_name}': +{captured} mau (tong: {sample_counts[class_id]})")
    return False


def main():
    labels = _load_labels()
    print("Cu chi duoc gan phim so:")
    for i, name in enumerate(labels):
        print(f"  [{i}] {name}")
    print(
        f"\nBam 1 phim so -> tu dong ghi lien tuc {config.BURST_CAPTURE_SECONDS:.0f} giay. "
        "GIU NGUYEN cu chi nhung CHU DONG di chuyen tay (xoay, nghieng, gan-xa, trai-phai) "
        "trong luc ghi de tao du lieu da dang. Nen ghi >= 5-6 dot/cu chi, doi kieu di chuyen "
        "va (neu duoc) ca anh sang/nen giua cac dot. Nhan 'q' de thoat.\n"
    )

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
            _draw_idle_overlay(display_frame, hand_landmarks, labels, sample_counts)
            cv2.imshow(_WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if hand_landmarks is not None and ord('0') <= key <= ord('9'):
                class_id = key - ord('0')
                if class_id < len(labels):
                    should_quit = _run_burst_capture(cap, tracker, class_id, labels[class_id], sample_counts)
                    if should_quit:
                        break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()