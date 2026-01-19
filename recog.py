import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
    QMessageBox, QProgressBar, QSizePolicy, QTextEdit
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# =================== 模型加载 ===================
emotion_model = YOLO(
    "D:/VScode/xiangmu/recognize-main/runs/train/fer2013_optimized/weights/best.pt"
)
face_model = YOLO(
    "D:/VScode/xiangmu/recognize-main/yolov8n-face.pt"
)

# =================== 表情映射 ===================
class_name_to_chinese = {
    "Angry": "Angry!",
    "Disgust": "Disgust!",
    "Fear": "Fear!",
    "Happy": "Happy!",
    "Sad": "Sad!",
    "Surprise": "Surprise!",
    "Neutral": "Neutral"
}

# =================== 中文绘制 ===================
def draw_text_chinese(img, text, pos,
                      font_path="simsun.ttc",
                      font_size=20,
                      color=(0, 255, 0)):
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# =================== 视频线程 ===================
class VideoThread(QThread):
    frame_signal = pyqtSignal(np.ndarray)
    progress_signal = pyqtSignal(int)
    result_signal = pyqtSignal(str)

    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.running = True
        self.paused = False
        self.interval = 6
        self.hold_frames = 8
        self.hold_counter = 0
        self.last_boxes = []

    def run(self):
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        count = 0

        while self.running and self.cap.isOpened():
            if self.paused:
                continue
            ret, frame = self.cap.read()
            if not ret:
                break
            count += 1
            self.progress_signal.emit(int(count / total * 100))

            if count % self.interval == 0:
                frame, texts, boxes = self.process_frame(frame)
                if boxes:
                    self.last_boxes = boxes
                    self.hold_counter = self.hold_frames
                self.result_signal.emit(texts)

            # 绘制缓存框
            if self.hold_counter > 0 and self.last_boxes:
                for (x1, y1, x2, y2, name, conf) in self.last_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (59,130,246), 2)
                    frame[:] = draw_text_chinese(frame, f"{name} {conf:.2f}", (x1, y1-28))
                self.hold_counter -= 1

            self.frame_signal.emit(frame)
            self.msleep(20)

        self.cap.release()
        self.progress_signal.emit(100)
        self.result_signal.emit("视频播放完毕！")

    def process_frame(self, frame):
        texts = []
        boxes = []
        h, w = frame.shape[:2]

        face_results = face_model(frame, conf=0.4)[0]
        if face_results.boxes is None:
            return frame, "", boxes

        for box in face_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb).resize((48,48))
            face_pil.save("temp_face.jpg")

            emo = emotion_model("temp_face.jpg")[0]

            if emo.boxes is None or len(emo.boxes.conf) == 0:
                continue

            best = int(np.argmax(emo.boxes.conf.cpu().numpy()))
            cls = int(emo.boxes.cls[best])
            conf = float(emo.boxes.conf[best])
            name = class_name_to_chinese.get(emotion_model.names[cls], "未知")

            boxes.append((x1, y1, x2, y2, name, conf))
            texts.append(f"{name}: {conf:.2f}")

        return frame, "\n".join(texts), boxes

    def toggle_pause(self):
        self.paused = not self.paused

# =================== 主窗口 ===================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸表情识别系统")
        self.setGeometry(100, 100, 1000, 700)
        self.image_path = None
        self.video_thread = None

        # ================== 样式 ==================
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e2f; font-family: 'Microsoft YaHei'; font-size: 14px; }
            QLabel { color: #ffffff; }
            QLabel#ImageLabel { background-color: #000000; border-radius: 14px; border: 1px solid #2f2f45; }
            QTextEdit { background-color: #2a2a3c; border: 1px solid #3a3a4f; border-radius: 10px; padding: 8px; color: #e6e6e6; }
            QPushButton { background-color: #3b82f6; color: white; border-radius: 8px; padding: 8px; font-weight: bold; }
            QPushButton:hover { background-color: #2563eb; }
            QPushButton:pressed { background-color: #1d4ed8; }
            QProgressBar { background-color: #2a2a3c; border-radius: 6px; text-align: center; color: white; }
            QProgressBar::chunk { background-color: #22c55e; border-radius: 6px; }
        """)

        # ================== 布局 ==================
        main = QWidget(self)
        self.setCentralWidget(main)
        layout = QHBoxLayout(main)

        # 左侧
        left = QVBoxLayout()
        self.image_label = QLabel("展示区域")
        self.image_label.setObjectName("ImageLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left.addWidget(self.image_label)
        self.progress = QProgressBar()
        left.addWidget(self.progress)

        # 右侧
        right = QVBoxLayout()
        self.result_label = QLabel("预测结果")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #38bdf8;")
        right.addWidget(self.result_label)
        self.result = QTextEdit()
        self.result.setReadOnly(True)
        self.result.setFixedWidth(260)
        right.addWidget(self.result)

        btns = QVBoxLayout()
        self.btn_img = QPushButton("选择图片")
        self.btn_img.clicked.connect(self.choose_image)
        btns.addWidget(self.btn_img)

        self.btn_detect = QPushButton("检测图片")
        self.btn_detect.clicked.connect(self.detect_image)
        btns.addWidget(self.btn_detect)

        self.btn_video = QPushButton("选择视频")
        self.btn_video.clicked.connect(self.choose_video)
        btns.addWidget(self.btn_video)

        self.btn_play = QPushButton("播放视频")
        self.btn_play.clicked.connect(self.play_video)
        btns.addWidget(self.btn_play)

        self.btn_pause = QPushButton("暂停 / 继续")
        self.btn_pause.clicked.connect(self.toggle_pause)
        btns.addWidget(self.btn_pause)

        right.addLayout(btns)
        layout.addLayout(left, 3)
        layout.addLayout(right, 1)

    # ================== 图片 ==================
    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.jpg *.png)")
        if path:
            self.image_path = path
            self.show_image(path)

    def detect_image(self):
        if not self.image_path:
            return
        img = cv2.imread(self.image_path)
        h, w = img.shape[:2]
        texts = []
        face_results = face_model(img, conf=0.4)[0]

        if face_results.boxes is not None:
            for box in face_results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                face = img[y1:y2, x1:x2]
                if face.size == 0:
                    continue
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_pil = Image.fromarray(face_rgb).resize((48,48))
                face_pil.save("temp_face.jpg")
                emo = emotion_model("temp_face.jpg")[0]
                if emo.boxes is None or len(emo.boxes.conf) == 0:
                    continue
                best = int(np.argmax(emo.boxes.conf.cpu().numpy()))
                cls = int(emo.boxes.cls[best])
                conf = float(emo.boxes.conf[best])
                name = class_name_to_chinese.get(emotion_model.names[cls], "未知")
                cv2.rectangle(img, (x1, y1), (x2, y2), (59,130,246),3)
                img = draw_text_chinese(img, f"{name} {conf:.2f}", (x1, y1-28))
                texts.append(f"{name}: {conf:.2f}")
        os.makedirs("outputs", exist_ok=True)
        out = "outputs/result.jpg"
        cv2.imwrite(out, img)
        self.show_image(out)
        self.result.setText("\n".join(texts))

    def show_image(self, path):
        pix = QPixmap(path).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pix)

    # ================== 视频 ==================
    def choose_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi)")

    def play_video(self):
        if not hasattr(self, "video_path") or not self.video_path:
            return
        self.video_thread = VideoThread(self.video_path)
        self.video_thread.frame_signal.connect(self.update_frame)
        self.video_thread.progress_signal.connect(self.progress.setValue)
        self.video_thread.result_signal.connect(self.result.setText)
        self.video_thread.start()

    def toggle_pause(self):
        if self.video_thread:
            self.video_thread.toggle_pause()

    def update_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

# ================== 程序入口 ==================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
