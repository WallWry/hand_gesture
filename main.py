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


def main():
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    ok, _ = cap.read()
    if not ok:
        raise RuntimeError("Khong mo duoc webcam (kiem tra quyen truy cap camera / index 0).")

    detector = HandGestureDetector()
    robot = RobotController()

    # Robot bat dau o trang thai DUNG cho toi khi nhan ro cu chi dau tien.
    robot.send(STOP)

    # Xac nhan cu chi qua nhieu frame lien tiep truoc khi doi lenh robot -
    # xem giai thich o config.GESTURE_CONFIRM_FRAMES.
    pending_gesture = None
    pending_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[Loi] Mat ket noi camera, dung chuong trinh.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gesture, hand_landmarks = detector.process(frame_rgb)

        # Frame "chua ro" (gesture is None - landmark rung nhe qua vung dead
        # zone giua 2 nguong) KHONG lam mat tien do dang dem - chi mot cu
        # chi TRAI NGUOC moi reset ve 0. Truoc day reset ca khi gap 1 frame
        # None don le, khien chuoi 5-frame lien tiep gan nhu khong bao gio
        # dat duoc trong dieu kien thuc te (landmark luon rung nhe).
        if gesture is not None:
            if gesture == pending_gesture:
                pending_count += 1
            else:
                pending_gesture = gesture
                pending_count = 1

            if pending_count >= config.GESTURE_CONFIRM_FRAMES:
                robot.send(_GESTURE_TO_COMMAND[pending_gesture])

        display_frame = cv2.flip(frame, 1) if config.MIRROR_DISPLAY else frame

        if hand_landmarks is not None:
            # Goc gap tung ngon - de xem TRUC TIEP tren man hinh khi tinh
            # chinh FINGER_STRAIGHT_ANGLE_DEG/FINGER_CURLED_ANGLE_DEG trong
            # config.py cho dung tay/khoang cach camera thuc te cua ban.
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

        status_text = f"Cu chi: {gesture or '...'} | Robot: {robot.state}"
        cv2.putText(display_frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Hand Gesture Robot Control (q de thoat)", display_frame)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
