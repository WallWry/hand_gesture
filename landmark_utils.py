"""Tien xu ly 21 landmark ban tay thanh 1 feature vector cho keypoint
classifier (train_keypoint_classifier.py / keypoint_classifier.py), theo
dung cach lam cua repo tham khao Kazuhito00/hand-gesture-recognition-using-
mediapipe (calc_landmark_list + pre_process_landmark trong app.py cua ho).

Quy trinh:
  1. Doi toa do landmark chuan hoa [0,1] cua MediaPipe ve toa do PIXEL thuc
     te theo kich thuoc khung hinh - BAT BUOC vi khung hinh khong vuong (vd
     1280x720): neu giu nguyen x,y chuan hoa [0,1], mot don vi theo truc x
     ung voi 1280px nhung theo truc y chi ung voi 720px, lam sai lech ty le
     hinh dang ban tay ma classifier hoc.
  2. Tru toa do co tay (landmark 0) khoi ca 21 diem -> toa do TUONG DOI so
     voi co tay, khong phu thuoc vi tri ban tay trong khung hinh
     (translation-invariant).
  3. Lam phang thanh vector 1 chieu (21 diem x 2 toa do x,y = 42 gia tri).
     Chi dung x,y, BO z: z do MediaPipe uoc luong tu 1 camera don (khong co
     depth that) nen nhieu/kem on dinh hon nhieu so voi x,y - dung z co the
     lam hai chat luong feature thay vi giup ich.
  4. Chuan hoa ve [-1, 1] bang cach chia ca vector cho gia tri tuyet doi
     lon nhat trong no -> scale-invariant (tay o gan/xa camera cho vector
     tuong tu nhau, khong can them nhieu du lieu o moi khoang cach khac
     nhau khi train).
"""


def landmarks_to_feature_vector(hand_landmarks, image_width, image_height):
    """Tra ve list 42 float (da chuan hoa [-1, 1]) dai dien hinh dang ban
    tay. hand_landmarks: list 21 NormalizedLandmark (co .x, .y trong [0,1])
    tu HandTracker.detect(). Dung chung cho ca luc train (collect_landmark_
    data.py) lan luc suy luan (gesture_detector.py) - PHAI giong het nhau,
    neu khong model se nhan input khac phan bo voi luc train."""
    pixel_points = [(lm.x * image_width, lm.y * image_height) for lm in hand_landmarks]

    wrist_x, wrist_y = pixel_points[0]
    relative_points = [(x - wrist_x, y - wrist_y) for x, y in pixel_points]

    flattened = [coord for point in relative_points for coord in point]

    max_abs = max((abs(v) for v in flattened), default=0.0)
    if max_abs < 1e-9:
        return flattened  # tranh chia 0 - khong nen xay ra voi ban tay that
    return [v / max_abs for v in flattened]
