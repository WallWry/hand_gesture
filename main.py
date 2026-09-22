"""
Dieu khien robot bang cu chi tay qua camera - happy case (2 cu chi):
  - Xoe ban tay (moi ngon duoi thang) -> robot TIEN (FORWARD)
  - Nam ban tay (nam dam)             -> robot DUNG (STOP)
Nhan dien on dinh voi nhieu huong xoay tay khac nhau - xem gesture_detector.py.

Truoc khi chay lan dau, tai model hand_landmarker.task (mot lan):
  curl -sSL -o hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

Chay: python main.py
Thoat: nhan 'q' trong cua so video.
"""

import cv2
import mediapipe as mp

import config
from gesture_detector import HandGestureDetector, OPEN_PALM, FIST, finger_angles, thumb_spread_ratio
from robot_controller import RobotController, FORWARD, STOP

_FINGER_NAMES = ["index", "middle", "ring", "pinky"]
_GESTURE_TO_COMMAND = {OPEN_PALM: FORWARD, FIST: STOP}
_DRAWING_UTILS = mp.tasks.vision.drawing_utils
_HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS


class GestureConfirmer:
    """Chi "chot" 1 cu chi sau khi nhan CUNG mot gesture du
    config.GESTURE_CONFIRM_FRAMES frame lien tiep - tranh giat lenh khi cu
    chi dang chuyen tiep giua 2 trang thai.

    Frame "chua ro" (gesture=None - landmark rung nhe qua vung dead zone
    giua 2 nguong) KHONG lam mat tien do dang dem, chi mot cu chi TRAI
    NGUOC moi reset ve 0 - neu khong, chuoi N-frame lien tiep se gan nhu
    khong bao gio dat duoc trong dieu kien thuc te (landmark luon rung nhe).
    """

    def __init__(self):
        self._pending_gesture = None
        self._pending_count = 0

    def update(self, gesture):
        """Tra ve gesture DA DUOC XAC NHAN (du so frame yeu cau), hoac None
        neu chua du hoac frame nay khong ro cu chi."""
        if gesture is None:
            return None
        if gesture == self._pending_gesture:
            self._pending_count += 1
        else:
            self._pending_gesture = gesture
            self._pending_count = 1
        return self._pending_gesture if self._pending_count >= config.GESTURE_CONFIRM_FRAMES else None

    def reset(self):
        self._pending_gesture = None
        self._pending_count = 0


def _open_camera():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    ok, _ = cap.read()
    if not ok:
        raise RuntimeError("Khong mo duoc webcam (kiem tra quyen truy cap camera / index 0).")
    return cap


def _read_frame_with_retry(cap):
    """Thu doc frame toi da MAX_CAMERA_READ_RETRIES lan - xem giai thich
    o config.MAX_CAMERA_READ_RETRIES. Tra ve None neu het luot ma van fail."""
    for _ in range(config.MAX_CAMERA_READ_RETRIES):
        ok, frame = cap.read()
        if ok:
            return frame
    return None


def _draw_overlay(display_frame, hand_landmarks, gesture, robot_state):
    if hand_landmarks is not None:
        # Goc gap tung ngon + ty le xoe ngon cai - hien TRUC TIEP tren man
        # hinh de tinh chinh FINGER_STRAIGHT_ANGLE_DEG/FINGER_CURLED_ANGLE_DEG/
        # THUMB_SPREAD_RATIO trong config.py cho dung tay/camera thuc te.
        angles = finger_angles(hand_landmarks)
        thumb_ratio = thumb_spread_ratio(hand_landmarks)

        if config.MIRROR_DISPLAY:
            # ve tren frame da lat -> phai lat x cua tung landmark de khop vi tri hien thi
            for lm in hand_landmarks:
                lm.x = 1.0 - lm.x
        _DRAWING_UTILS.draw_landmarks(display_frame, hand_landmarks, _HAND_CONNECTIONS)

        debug_text = " ".join(f"{name}:{a:.0f}" for name, a in zip(_FINGER_NAMES, angles))
        debug_text += f" thumb_ratio:{thumb_ratio:.2f}"
        cv2.putText(display_frame, debug_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 1, cv2.LINE_AA)

    status_text = f"Cu chi: {gesture or '...'} | Robot: {robot_state}"
    cv2.putText(display_frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def main():
    cap = _open_camera()
    detector = HandGestureDetector()
    robot = RobotController()
    confirmer = GestureConfirmer()

    robot.send(STOP)  # trang thai an toan mac dinh cho toi khi nhan ro cu chi dau tien
    no_hand_frames = 0

    try:
        while True:
            frame = _read_frame_with_retry(cap)
            if frame is None:
                print("[Loi] Mat ket noi camera qua lau, dung chuong trinh.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gesture, hand_landmarks = detector.process(frame_rgb)

            # An toan: MAT HAN tay (khac voi "thay tay nhung cu chi mo ho")
            # qua lau -> robot phai tu dung, khong duoc giu nguyen lenh cu -
            # xem config.NO_HAND_STOP_FRAMES. Reset ca bo dem xac nhan cu
            # chi de khi tay quay lai, robot can 1 chuoi xac nhan MOI truoc
            # khi tien tiep, khong tu dong chay lai ngay lap tuc.
            if hand_landmarks is None:
                no_hand_frames += 1
                if no_hand_frames >= config.NO_HAND_STOP_FRAMES:
                    robot.send(STOP)
                    confirmer.reset()
            else:
                no_hand_frames = 0

            confirmed_gesture = confirmer.update(gesture)
            if confirmed_gesture is not None:
                robot.send(_GESTURE_TO_COMMAND[confirmed_gesture])

            display_frame = cv2.flip(frame, 1) if config.MIRROR_DISPLAY else frame
            _draw_overlay(display_frame, hand_landmarks, gesture, robot.state)

            cv2.imshow("Hand Gesture Robot Control (q de thoat)", display_frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    finally:
        # Luon giai phong camera/model du vong lap ket thuc binh thuong
        # ('q') hay do loi/exception giua chung - tranh giu treo camera.
        detector.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
