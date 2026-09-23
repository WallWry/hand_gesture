# Hand Gesture Robot Control

Điều khiển robot di chuyển bằng cử chỉ tay qua camera. Hệ thống định vị 21
điểm landmark bàn tay bằng MediaPipe, sau đó dùng một **classifier tự
train** (không dùng model cử chỉ có sẵn) để nhận diện cử chỉ và gửi lệnh
điều khiển.

Happy case hiện tại — 2 cử chỉ:

| Cử chỉ | Hình | Lệnh robot |
|---|---|---|
| Xòe bàn tay (`OPEN_PALM`) | 5 ngón duỗi thẳng | `FORWARD` (tiến) |
| Nắm bàn tay (`FIST`) | Nắm đấm | `STOP` (dừng) |

## Yêu cầu hệ thống

- Python 3.9 trở lên (đã test với 3.13)
- Webcam
- Windows / macOS / Linux

## 1. Cài đặt

```bash
git clone https://github.com/WallWry/hand_gesture.git
cd hand_gesture_robot_control

# (khuyến nghị) tạo virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

## 2. Thiết lập lần đầu (bắt buộc)

Project **không đi kèm sẵn** model landmark, dữ liệu cử chỉ hay classifier
đã train (các file này bị `.gitignore` vì gắn với tay/ánh sáng/camera của
từng người dùng cụ thể) — mỗi người chạy 3 bước sau đúng 1 lần trước khi
dùng `main.py`:

### Bước 1 — Tải model định vị landmark (model có sẵn của Google)

Đây là model **định vị 21 khớp tay**, không phải model phân loại cử chỉ
(bước đó tự train ở bước 3), nên tải về dùng chung là an toàn:

```bash
curl -sSL -o hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
```

### Bước 2 — Thu dữ liệu cử chỉ thật của chính bạn

```bash
python collect_landmark_data.py
```

Đưa tay vào khung hình, bấm phím `0` (OPEN_PALM) hoặc `1` (FIST) — mỗi lần
bấm sẽ tự động ghi liên tục ~4 giây, trong lúc đó **giữ nguyên cử chỉ
nhưng chủ động xoay/nghiêng/tiến-lùi tay** để dữ liệu đa dạng. Nên ghi
5-6 đợt cho mỗi cử chỉ. Nhấn `q` để thoát. Xem thêm chi tiết trong
docstring đầu file `collect_landmark_data.py`.

### Bước 3 — Train classifier từ dữ liệu vừa thu

```bash
python train_keypoint_classifier.py
```

In ra accuracy/precision/recall/confusion matrix trên tập kiểm tra, lưu
model vào `model/keypoint_classifier.pkl`. Nếu độ chính xác thấp, quay
lại Bước 2 thu thêm dữ liệu (dữ liệu cũ không mất, chỉ ghi thêm) rồi train
lại.

## 3. Chạy chương trình

```bash
python main.py
```

- Cửa sổ camera hiện landmark bàn tay + cử chỉ nhận diện + độ tin cậy +
  trạng thái robot hiện tại.
- Nhấn `q` để thoát.
- Nếu không thấy tay quá `NO_HAND_STOP_FRAMES` frame liên tiếp, robot tự
  động `STOP` (an toàn khi mất tín hiệu điều khiển).

## Cấu trúc project

| File | Vai trò |
|---|---|
| `main.py` | Vòng lặp chính: đọc camera → nhận diện cử chỉ → gửi lệnh robot |
| `hand_tracker.py` | Định vị 21 landmark bàn tay (MediaPipe HandLandmarker) |
| `landmark_utils.py` | Chuẩn hoá landmark thành feature vector 42 chiều |
| `gesture_detector.py` | Ghép `HandTracker` + `KeypointClassifier` để nhận diện cử chỉ |
| `keypoint_classifier.py` | Nạp model đã train, suy luận cử chỉ |
| `collect_landmark_data.py` | Công cụ thu dữ liệu landmark thật (chạy 1 lần đầu) |
| `train_keypoint_classifier.py` | Train `RandomForestClassifier` từ dữ liệu thu được |
| `robot_controller.py` | Interface gửi lệnh robot (hiện chỉ in ra console, xem bên dưới) |
| `config.py` | Toàn bộ tham số cấu hình |
| `data/keypoint.csv` | Dữ liệu landmark đã thu (tự sinh, không commit) |
| `model/keypoint_classifier.pkl` | Model đã train (tự sinh, không commit) |
| `model/keypoint_classifier_label.csv` | Danh sách tên cử chỉ theo class id (có commit) |

## Cấu hình đáng chú ý (`config.py`)

- `CAMERA_INDEX` — đổi nếu máy có nhiều camera và mở nhầm cam.
- `CLASSIFIER_CONFIDENCE_THRESHOLD` (mặc định 0.8) — độ tin cậy tối thiểu
  để chấp nhận 1 dự đoán cử chỉ; hạ xuống nếu model "nhút nhát" (từ chối
  quá nhiều), tăng lên nếu robot phản ứng nhầm với cử chỉ mơ hồ.
- `GESTURE_CONFIRM_FRAMES` — số frame liên tiếp cùng cử chỉ trước khi thực
  sự gửi lệnh, tránh giật lệnh khi tay đang chuyển động.
- `NO_HAND_STOP_FRAMES` — số frame mất tay liên tiếp trước khi robot tự
  dừng an toàn.

## Kết nối robot thật

`robot_controller.py` hiện **chưa kết nối robot thật** (Omron LD-90), chỉ
in lệnh `FORWARD`/`STOP` ra console để dev/test không cần robot. Khi có
API/SDK thật của robot (HTTP REST, ROS Twist message...), chỉ cần sửa 2
hàm `_send_forward()` và `_send_stop()` trong file này — phần còn lại của
chương trình không cần đổi gì.

## Xử lý sự cố

| Vấn đề | Cách xử lý |
|---|---|
| `RuntimeError: Khong mo duoc webcam` | Kiểm tra quyền truy cập camera của hệ điều hành, hoặc đổi `CAMERA_INDEX` trong `config.py` |
| `RuntimeError: Khong tim thay model '...'` khi chạy `main.py` | Chưa chạy Bước 2 + Bước 3 ở trên |
| Accuracy thấp sau khi train | Thu thêm dữ liệu đa dạng hơn (nhiều góc xoay/khoảng cách/ánh sáng) rồi train lại, xem hướng dẫn trong `collect_landmark_data.py` |
| Robot phản ứng chậm/giật khi đổi cử chỉ | Tăng `GESTURE_CONFIRM_FRAMES` trong `config.py` |
