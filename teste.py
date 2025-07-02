import sys
import numpy as np
import librosa
import sounddevice as sd
import tensorflow as tf
from joblib import load
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QGraphicsDropShadowEffect
from PyQt5.QtCore import Qt, QTimer, QRectF
from PyQt5.QtGui import QFont, QColor, QPainter, QBrush, QPen

DURATION = 5
SR = 44100
N_MFCC = 13

# Carregar modelo e preprocessadores
model = tf.keras.models.load_model("./models/audio_nn_model.keras")
scaler = load("./models/scaler.joblib")
le = load("./models/label_encoder.joblib")

def extrair_features(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    delta_mean = np.mean(delta, axis=1)
    delta2_mean = np.mean(delta2, axis=1)
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    rms = np.mean(librosa.feature.rms(y=y))
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    return np.concatenate([
        mfcc_mean, delta_mean, delta2_mean,
        [spec_centroid], [spec_bw], contrast, [rolloff], [zcr], [rms], chroma
    ])

class PulseWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.radius = 100
        self.opacity = 0.2
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.animating = False
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def start(self):
        self.animating = True
        self.timer.start(50)

    def stop(self):
        self.animating = False
        self.timer.stop()
        self.radius = 100
        self.opacity = 0.2
        self.update()

    def animate(self):
        self.radius += 4
        self.opacity -= 0.015
        if self.opacity <= 0:
            self.radius = 100
            self.opacity = 0.2
        self.update()

    def paintEvent(self, event):
        if not self.animating:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        center = self.rect().center()
        brush = QBrush(QColor(63, 81, 181, int(self.opacity * 255)))
        painter.setBrush(brush)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, self.radius, self.radius)

class AudioClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Classificador de Áudio")
        self.setFixedSize(400, 450)
        self.setStyleSheet("background-color: #1a1a1a; color: white;")

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel("Clique para escutar")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Arial", 18))

        self.button = QPushButton("🔊")
        self.button.setFont(QFont("Arial", 48))
        self.button.setFixedSize(200, 200)
        self.button.setStyleSheet("border-radius: 100px; background-color: #3f51b5; color: white;")
        self.button.clicked.connect(self.gravar_e_classificar)

        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setXOffset(0)
        shadow.setYOffset(0)
        shadow.setColor(QColor(63, 81, 181))
        self.button.setGraphicsEffect(shadow)

        self.pulse = PulseWidget(self)
        self.pulse.setGeometry(QRectF(self.width()/2 - 150, self.height()/2 - 220, 300, 300).toRect())
        self.pulse.lower()

        self.layout.addStretch()
        self.layout.addWidget(self.button, alignment=Qt.AlignCenter)
        self.layout.addWidget(self.label, alignment=Qt.AlignCenter)
        self.layout.addStretch()

    def gravar_e_classificar(self):
        self.label.setText("🎙️ Gravando...")
        self.pulse.start()
        QApplication.processEvents()

        recording = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
        sd.wait()
        audio = recording.flatten()

        self.pulse.stop()

        try:
            features = extrair_features(audio, SR)
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)
            classe = le.inverse_transform([np.argmax(prediction)])[0]
            self.label.setText(f"🔍 Classe: {classe}")
        except Exception as e:
            self.label.setText("❌ Erro ao classificar")
            print("Erro:", e)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    janela = AudioClassifierApp()
    janela.show()
    sys.exit(app.exec_())
