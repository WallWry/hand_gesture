"""Nhan dien cu chi tay tu 21 diem landmark (MediaPipe Tasks HandLandmarker).

Ly do KHONG dung "GestureRecognizer" co san cua MediaPipe (nhu ban dau):
bo phan loai canned gesture ("Closed_Fist"/"Open_Palm") duoc train chu yeu
tren anh tay chinh dien, huong len - khi xoay long ban tay xuong duoi hoac
sang trai/phai, diem tin cay cua no tut duoi nguong du landmark van dinh vi
dung vi tri 21 khop, dan den gesture bi bo qua va robot khong phan ung.

Thay vao do, ta tu phan loai xoe/nam TRUC TIEP tu 21 diem landmark bang goc
gap tai khop PIP cua tung ngon (xem classify_gesture/_finger_states) - chi
so hinh hoc chuan de do "ngon thang hay cong", khong phu thuoc huong tay
trong anh (rotation-invariant) vi chi dung 3 diem CUA CHINH NGON DO (mcp,
pip, tip), khong tham chieu toi co tay hay huong tong the cua ban tay.

Khi nam tay, cac ngon thuong che khuat lan nhau tuy goc quay camera khien
landmark 1 ngon (thuong la pinky/ring) bi doan nhieu - vi vay FIST duoc
chap nhan khi >= MIN_CURLED_FOR_FIST/4 ngon cong lai (khong doi hoi ca 4),
de chiu duoc 1 ngon bi nhan sai. Nhung dung sai nay CHI ap dung khi ngon
con lai dang o trang thai "chua ro" (do nhieu/che khuat that su) - neu no
duoc do ro rang la DUOI THANG (vd chi 1 ngon tro ra co chu dich) thi KHONG
tinh la FIST nua (xem dieu kien straight_count == 0 trong classify_gesture),
tranh nham cu chi "gio 1 ngon" thanh nam tay.

Ngon cai duoc xu ly RIENG (khong dung goc gap PIP nhu 4 ngon kia, vi huong
gap cua no khac han - xem docstring _thumb_spread_ratio): dung de phan biet
"xoe du 5 ngon" (OPEN_PALM that su) voi "gio 4 ngon, cai thu vao" (truoc day
2 truong hop nay giong het nhau vi ngon cai khong duoc xet toi).
"""

import math

import mediapipe as mp

import config

OPEN_PALM = "OPEN_PALM"
FIST = "FIST"

# 4 ngon dung de xet (bo ngon cai - huong gap cua ngon cai khac han 4 ngon
# con lai, khong phu hop voi cung 1 nguong goc don gian nay).
# Moi ngon can 3 khop: mcp (goc ngon), pip (khop giua - noi tinh goc gap), tip (dau ngon).
_FINGER_JOINTS = [
    (5, 6, 8),     # index: mcp, pip, tip
    (9, 10, 12),   # middle
    (13, 14, 16),  # ring
    (17, 18, 20),  # pinky
]


class HandGestureDetector:
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

    def process(self, frame_rgb):
        """Tra ve (gesture, hand_landmarks) - hand_landmarks la None neu khong thay tay.

        gesture la OPEN_PALM/FIST/None; hand_landmarks la list NormalizedLandmark
        (dung de ve overlay), lay tay dau tien (happy case: 1 tay, xem config.NUM_HANDS).
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._frame_idx += 1
        result = self._landmarker.detect_for_video(mp_image, self._frame_idx)

        if not result.hand_landmarks:
            return None, None

        hand_landmarks = result.hand_landmarks[0]
        gesture = classify_gesture(hand_landmarks)
        return gesture, hand_landmarks

    def close(self):
        self._landmarker.close()


def _angle_deg(a, b, c):
    """Goc (do) tai dinh b, tao boi 2 vector b->a va b->c.

    Ngon duoi thang: 3 khop mcp-pip-tip gan nhu thang hang -> goc ~180.
    Ngon cong lai: pip gap manh -> goc nho di ro ret (thuong < 90 khi nam
    chat). Chi dung 3 diem cua rieng ngon do nen khong phu thuoc vi tri/
    huong cua co tay hay ca ban tay trong khung hinh.
    """
    v1 = (a.x - b.x, a.y - b.y, a.z - b.z)
    v2 = (c.x - b.x, c.y - b.y, c.z - b.z)
    dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    n1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    n2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
    cos_angle = dot / max(n1 * n2, 1e-9)
    cos_angle = max(-1.0, min(1.0, cos_angle))  # tranh loi lam tron vuot [-1, 1] cua acos
    return math.degrees(math.acos(cos_angle))


def finger_angles(landmarks):
    """Goc gap tai PIP cua 4 ngon index/middle/ring/pinky, theo dung thu tu do."""
    return [_angle_deg(landmarks[mcp], landmarks[pip], landmarks[tip])
            for mcp, pip, tip in _FINGER_JOINTS]


def _dist3(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def thumb_spread_ratio(landmarks):
    """Do "xoe ra" cua ngon cai = khoang cach (dau ngon cai -> goc ngon tro)
    chia cho kich thuoc long ban tay (co tay -> goc ngon giua, gan nhu
    khong doi bat ke ngon nao dang duoi/cong, dung lam thuoc do chuan hoa).

    Ngon cai KHONG dung goc gap PIP nhu 4 ngon kia duoc: no gap/xoe theo
    kieu doi chieu (opposable) chu khong cong vao long ban tay theo 1 mat
    phang don gian, nen goc tai khop cua no khong phan anh ro "xoe hay
    thu" nhu 4 ngon con lai. Khoang cach tu dau ngon cai toi goc ngon tro
    la chi so de phan biet: xoe het co (ban tay mo hoan toan, ty le lon)
    voi thu vao long ban tay/sat cac ngon khac (ty le nho, vd khi nam tay
    hoac khi chi gio 4 ngon con lai va giau ngon cai)."""
    palm_size = _dist3(landmarks[0], landmarks[9])  # wrist -> middle_mcp
    thumb_to_index_mcp = _dist3(landmarks[4], landmarks[5])
    return thumb_to_index_mcp / max(palm_size, 1e-6)


def classify_gesture(landmarks):
    """OPEN_PALM: ca 4 ngon (index/middle/ring/pinky) deu thang VA ngon cai
    cung xoe ra (khong thi chi la "gio 4 ngon, giau ngon cai" - de bi nham
    voi xoe tay that neu khong kiem tra ngon cai).

    FIST: >= MIN_CURLED_FOR_FIST/4 ngon cong lai, VA khong ngon nao trong
    so con lai duoc do ro rang la DUOI THANG (straight_count == 0) - dieu
    kien thu 2 nay chan truong hop "gio 1 ngon co chu dich" (vd chi tay)
    vo tinh roi vao dung sai "chiu duoc 1 ngon nhieu" danh cho fist that.
    """
    angles = finger_angles(landmarks)
    straight_count = sum(1 for a in angles if a >= config.FINGER_STRAIGHT_ANGLE_DEG)
    curled_count = sum(1 for a in angles if a <= config.FINGER_CURLED_ANGLE_DEG)
    thumb_spread = thumb_spread_ratio(landmarks) >= config.THUMB_SPREAD_RATIO

    if straight_count >= config.MIN_STRAIGHT_FOR_OPEN_PALM and thumb_spread:
        return OPEN_PALM
    if curled_count >= config.MIN_CURLED_FOR_FIST and straight_count == 0:
        return FIST
    return None  # trang thai trung gian - chua ro cu chi
