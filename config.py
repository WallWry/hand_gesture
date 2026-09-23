"""
Cau hinh cho dieu khien robot bang cu chi tay qua camera.
Happy case: chi 2 cu chi - xoe ban tay (FORWARD) va nam tay (STOP).
"""

# ==== Capture ====
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CAMERA_INDEX = 0

# Chi lat anh khi HIEN THI cho de nhin (giong soi guong). Khong anh huong
# toa do landmark dung de nhan dien cu chi - luon tinh tren frame goc.
MIRROR_DISPLAY = True

# ==== MediaPipe Tasks - HandLandmarker ====
# Chi dung model dinh vi 21 diem landmark (khong dung bo phan loai cu chi
# co san "GestureRecognizer" cua Google - bo do duoc train chu yeu tren tu
# the tay chinh dien nen do tin cay tut manh khi xoay tay xuong/sang trai-
# phai, du landmark van dinh vi dung. Buoc PHAN LOAI cu chi tu 21 toa do do
# la mot classifier TU TRAIN, xem KEYPOINT_* ben duoi va gesture_detector.py).
HAND_MODEL_PATH = "hand_landmarker.task"
NUM_HANDS = 1  # happy case: chi 1 tay dieu khien
MIN_HAND_DETECTION_CONFIDENCE = 0.7
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# ==== Keypoint classifier (phan loai cu chi tu landmark) ====
# Toan bo pipeline: collect_landmark_data.py (thu du lieu that) ->
# train_keypoint_classifier.py (train + luu model) ->
# keypoint_classifier.py (nap model, suy luan) -> gesture_detector.py (ghep
# voi HandTracker). Thay the hoan toan cach lam cu la tu dat nguong goc
# gap/ty le ngon cai thu cong (xem lich su git) - danh doi 1 lan cong suc
# thu du lieu de model tu hoc duong bien phan loai, thay vi phai doan/tinh
# chinh tung nguong tay cho tung dieu kien anh sang/goc camera.
KEYPOINT_DATASET_PATH = "data/keypoint.csv"
KEYPOINT_MODEL_PATH = "model/keypoint_classifier.pkl"
KEYPOINT_LABEL_PATH = "model/keypoint_classifier_label.csv"

# Do tin cay toi thieu (xac suat lop du doan cao nhat tu predict_proba) de
# CHAP NHAN 1 du doan - duoi nguong nay coi la "chua ro cu chi" (tra ve
# None), tuong duong vung "dead zone" giua 2 nguong goc cua cach lam cu.
# Ha xuong neu model tu choi qua nhieu frame ro rang dung cu chi (model
# "nhut nhat"); tang len neu robot phan ung nham voi cu chi mo ho/dang
# chuyen tiep giua 2 trang thai.
CLASSIFIER_CONFIDENCE_THRESHOLD = 0.8

# ==== Thu du lieu (collect_landmark_data.py) ====
# Moi lan bam 1 phim so, script GHI LIEN TUC trong BURST_CAPTURE_SECONDS
# giay thay vi chi ghi 1 frame tinh - nguoi thu giu nguyen cu chi nhung CHU
# DONG di chuyen tay (xoay/nghieng/gan-xa/trai-phai) trong luc do, nho vay
# 1 lan bam tao ra hang chuc mau DA DANG thay vi phai bam tay hang tram
# lan. Xem huong dan chi tiet trong docstring collect_landmark_data.py.
BURST_CAPTURE_SECONDS = 4.0

# Chi ghi 1 frame moi BURST_SAMPLE_EVERY_N_FRAMES frame trong luc ghi lien
# tuc (khong ghi ca 30fps) - tranh qua nhieu mau GAN TRUNG NHAU (2 frame
# lien tiep o 30fps gan nhu giong het nhau, khong them thong tin gi cho
# model ma chi lam file du lieu phinh to). ~10 mau/giay o 30fps la du day
# ma van bat duoc chuyen dong cua tay trong luc ghi.
BURST_SAMPLE_EVERY_N_FRAMES = 3

# So frame LIEN TIEP phai cung mot cu chi truoc khi thuc su gui lenh cho
# robot - tranh giat lenh khi cu chi dang chuyen tiep giua 2 trang thai.
GESTURE_CONFIRM_FRAMES = 5

# ==== An toan ====
# So frame LIEN TIEP khong thay tay nao (khac voi "thay tay nhung cu chi
# mo ho") truoc khi ROBOT TU DUNG, bat ke lenh truoc do la gi. Day la co
# che fail-safe: neu dang FORWARD ma nguoi dieu khien buoc ra khoi khung
# hinh/che khuat hoan toan tay, khong duoc de robot "giu nguyen lenh cu"
# va tiep tuc tien vo thoi han - phai chu dong dung lai khi mat tin hieu
# dieu khien. ~15 frame (~0.5-1s o 15-30fps) - du ngan de an toan, du dai
# de khong dung nham vi 1-2 frame mat tay thoang qua (rung tay, chop mat
# webcam).
NO_HAND_STOP_FRAMES = 15

# So lan doc frame tu webcam that bai LIEN TIEP truoc khi dung han chuong
# trinh (thay vi dung ngay o lan doc that bai dau tien - webcam USB thinh
# thoang drop 1 frame la binh thuong, khong nen lam robot mat hoan toan
# tin hieu dieu khien chi vi 1 frame loi thoang qua). Cung gia tri va ly
# do nhu MAX_CAMERA_READ_RETRIES trong camera_distance_estimation.
MAX_CAMERA_READ_RETRIES = 30