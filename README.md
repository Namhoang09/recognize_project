🎤👤 Voice & Face Recognition System (MLP + FaceID + Web)
📌 Giới thiệu

Dự án này xây dựng hệ thống nhận diện giọng nói và khuôn mặt kết hợp, sử dụng:

🎤 Voice Recognition: Mô hình MLP (Multi-Layer Perceptron)

👤 Face Recognition: Dựa trên encoding (kiểu FaceID)

🌐 Web Interface: Hiển thị kết quả realtime qua web (Flask + SocketIO)

Ứng dụng có thể:

Nhận diện người dùng qua giọng nói

Nhận diện người dùng qua khuôn mặt

Gửi kết quả lên giao diện web theo thời gian thực

⚙️ Công nghệ sử dụng

Python

Flask + Flask-SocketIO

NumPy

OpenCV

face_recognition

SoundDevice / Librosa

Scikit-learn (MLP)

📂 Cấu trúc thư mục
RECOGNIZE_PROJECT/
│
├── dataset/
│   ├── dataset_faces/      # Dữ liệu ảnh khuôn mặt
│   └── dataset_voices/     # Dữ liệu âm thanh
│
├── model/
│   ├── face_model/         # Model nhận diện khuôn mặt
│   └── voice_model/        # Model MLP cho giọng nói
│
├── train/
│   ├── face_train/         # Code train face
│   └── voice_train/        # Code train voice
│
├── templates/
│   └── index.html          # Giao diện web
│
├── testing/                # Kết quả đánh giá
│
├── main.py                 # Chạy nhận diện
├── server_app.py           # Flask server
├── gemini_bot.py           # (tuỳ chọn AI interaction)
├── system_utils.py         # Utility functions
├── requirements.txt
└── README.md

🚀 Cách chạy project
1. Cài đặt thư viện
pip install -r requirements.txt

2. Train model (nếu chưa có)
🎤 Voice
python train/voice_train/train.py

👤 Face
python train/face_train/train.py

3. Chạy server
python server_app.py

4. Chạy nhận diện
python main.py

5. Mở web

Truy cập:

http://localhost:5000

🧠 Cách hoạt động
🎤 Voice Recognition

Thu âm từ microphone

Chia thành các đoạn (window)

Loại bỏ silence bằng threshold

Trích xuất feature

Dự đoán bằng MLP

👤 Face Recognition

Capture từ camera

Encode khuôn mặt

So sánh với dataset

Trả về tên người

🌐 Web

Dữ liệu gửi qua SocketIO

Hiển thị realtime:

Tên

Confidence

Trạng thái nhận diện

📊 Kết quả

Confusion Matrix

Accuracy

FPS realtime

(Xem trong thư mục testing/)

📌 Tính năng nổi bật

Nhận diện đa modal (voice + face)

Realtime streaming

Dễ mở rộng (AI chatbot, security system,…)

Code tách module rõ ràng

🔧 Hướng phát triển

Kết hợp voice + face để tăng độ chính xác

Thêm đăng nhập bảo mật

Deploy lên server / cloud

Tối ưu latency realtime
