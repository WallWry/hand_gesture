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
# co san "GestureRecognizer" cua Google nua - bo do duoc train chu yeu tren
# tu the tay chinh dien nen do tin cay tut manh khi xoay tay xuong/sang
# trai-phai, du landmark van dinh vi dung. Ta tu phan loai xoe/nam tu toa
# do landmark trong gesture_detector.classify_gesture(), on dinh hon nhieu
# khi tay xoay theo moi huong - xem giai thich chi tiet trong ham do).
HAND_MODEL_PATH = "hand_landmarker.task"
NUM_HANDS = 1  # happy case: chi 1 tay dieu khien
MIN_HAND_DETECTION_CONFIDENCE = 0.7
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# Goc gap (do) tai khop PIP cua tung ngon - xem gesture_detector._angle_deg.
# >= FINGER_STRAIGHT_ANGLE_DEG -> ngon duoc coi la "duoi thang".
# <= FINGER_CURLED_ANGLE_DEG   -> ngon duoc coi la "cong lai".
# Khoang giua 2 nguong (vd 100-160 do) la vung "chua ro", khong tinh cho
# ben nao - tranh nhay lat qua lai khi ngon o trang thai nua chung.
FINGER_STRAIGHT_ANGLE_DEG = 160.0
FINGER_CURLED_ANGLE_DEG = 100.0

# So ngon (trong 4 ngon xet) toi thieu phai o dung trang thai de chot cu
# chi. OPEN_PALM van doi ca 4/4 thang. FIST chi doi 3/4 cong lai - khi nam
# tay, 1 ngon (thuong la pinky/ring) hay bi ngon khac che khuat tuy goc
# camera khien landmark cua no bi doan nhieu/sai; doi ca 4/4 se de mat tin
# hieu STOP chi vi 1 ngon nhieu (dung sai nay chi ap dung khi ngon con lai
# THAT SU chua ro - xem dieu kien straight_count == 0 trong classify_gesture,
# chan truong hop gio 1 ngon co chu dich bi nham thanh fist).
MIN_STRAIGHT_FOR_OPEN_PALM = 4
MIN_CURLED_FOR_FIST = 3

# Ty le (khoang cach dau-ngon-cai toi goc-ngon-tro) / (kich thuoc long ban
# tay) toi thieu de coi ngon cai la "xoe ra" - xem
# gesture_detector.thumb_spread_ratio(). Bat buoc them dieu kien nay cho
# OPEN_PALM de phan biet voi "gio 4 ngon, giau ngon cai" (truoc day 2
# truong hop nay giong het nhau vi ngon cai khong duoc xet).
THUMB_SPREAD_RATIO = 0.5

# So frame LIEN TIEP phai cung mot cu chi truoc khi thuc su gui lenh cho
# robot - tranh giat lenh khi cu chi dang chuyen tiep giua 2 trang thai.
GESTURE_CONFIRM_FRAMES = 5
