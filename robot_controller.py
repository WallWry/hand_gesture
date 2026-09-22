"""Interface gui lenh cho robot.

Chua co ket noi toi robot that (Omron LD-90) trong moi truong dev hien tai
(tuong tu camera_distance_estimation - chua co camera that nen dung webcam
thay the). RobotController hien chi in lenh ra console; khi co API/SDK that
cua robot (vd HTTP REST, ROS twist message...), thay noi dung 2 ham
_send_forward()/_send_stop() ben duoi bang lenh goi thuc te, phan con lai
cua chuong trinh (main.py) khong can doi gi.
"""

FORWARD = "FORWARD"
STOP = "STOP"


class RobotController:
    def __init__(self):
        self._state = None  # lenh hien tai robot dang giu, de tranh gui lap lenh giong het frame truoc

    @property
    def state(self):
        return self._state

    def send(self, command):
        if command == self._state:
            return  # da o dung trang thai nay roi, khong can gui lai
        if command == FORWARD:
            self._send_forward()
        elif command == STOP:
            self._send_stop()
        else:
            raise ValueError(f"Lenh khong hop le: {command}")
        self._state = command

    def _send_forward(self):
        print("[ROBOT] -> FORWARD")

    def _send_stop(self):
        print("[ROBOT] -> STOP")
