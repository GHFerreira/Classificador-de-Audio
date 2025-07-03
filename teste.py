import sys
import numpy as np
import sounddevice as sd
import librosa
import tensorflow as tf
from joblib import load
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QPainter, QColor, QPen, QIcon, QFont
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QGraphicsDropShadowEffect


DURATION = 5
SR = 44100
N_MFCC = 13


class PulseBackground(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Classificador de Áudio")
        self.setGeometry(100, 100, 500, 500)
        self.setStyleSheet("background-color: #121212;")
        self.pulses = []

        self.mic_level = 0.0

        self.model = tf.keras.models.load_model("./models/audio_nn_model.keras")
        self.scaler = load("./models/scaler.joblib")
        self.le = load("./models/label_encoder.joblib")

        self.recording = None
        self.is_recording = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)

        self.spawn_timer = QTimer()
        self.spawn_timer.timeout.connect(self.spawn_pulse)

        # Botão circular central com ícone PNG
        self.button = QPushButton("", self)
        self.button.setIcon(QIcon("./icone.png"))
        self.button.setIconSize(QSize(80, 80))
        self.button.setFixedSize(100, 100)
        self.button.setStyleSheet("""
            QPushButton {
                background-color: #2c387e;
                border-radius: 50px;
                border: none;
            }
            QPushButton:pressed {
                background-color: #1a237e;
            }
        """)
        self.button.clicked.connect(self.iniciar_animacao)

        # Label para mensagem abaixo do botão, com fundo transparente
        self.status_label = QLabel("", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedWidth(200)
        self.status_label.setStyleSheet("color: white; font-size: 16px; background: transparent;")
        self.status_label.setAttribute(Qt.WA_TranslucentBackground)

        # Sombra no botão
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 160))
        self.button.setGraphicsEffect(shadow)

        # Stream áudio para volume animado
        self.chunk_size = 1024
        self.stream = sd.InputStream(callback=self.audio_callback, channels=1, samplerate=SR, blocksize=self.chunk_size)
        self.stream.start()

        self.center_widgets()

    def center_widgets(self):
        # Centraliza botão
        self.button.move(self.width() // 2 - self.button.width() // 2,
                         self.height() // 2 - self.button.height() // 2)

        # Posiciona label logo abaixo do botão, centralizada
        self.status_label.move(self.width() // 2 - self.status_label.width() // 2,
                               self.button.y() + self.button.height() + 10)

    def resizeEvent(self, event):
        self.center_widgets()

    def iniciar_animacao(self):
        if self.is_recording:
            return
        self.is_recording = True
        self.pulses.clear()
        self.timer.start(40)
        self.spawn_timer.start(700)
        self.button.setEnabled(False)
        self.status_label.setText("🎙️ Gravando...")

        self.recording = sd.rec(int(DURATION * SR), samplerate=SR, channels=1)
        QTimer.singleShot(DURATION * 1000, self.finalizar_gravacao)

    def finalizar_gravacao(self):
        sd.wait()
        self.spawn_timer.stop()
        self.timer.stop()

        audio = self.recording.flatten()
        try:
            features = self.extrair_features(audio, SR)
            features_scaled = self.scaler.transform([features])
            prediction = self.model.predict(features_scaled)
            classe = self.le.inverse_transform([np.argmax(prediction)])[0]
            self.status_label.setText(f"🔍 {classe}")
        except Exception as e:
            self.status_label.setText("❌ Erro")
            print("Erro na classificação:", e)

        self.is_recording = False
        self.button.setEnabled(True)

        # Resetar animação para sumir as bolhas
        self.pulses.clear()
        self.update()

    def extrair_features(self, y, sr):
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

    def spawn_pulse(self):
        self.pulses.append({
            'radius': 50,
            'opacity': 1.0,
            'width': 2
        })

    def update_animation(self):
        new_pulses = []
        for pulse in self.pulses:
            pulse['radius'] += 5
            pulse['opacity'] -= 0.02
            if pulse['opacity'] > 0:
                new_pulses.append(pulse)
        self.pulses = new_pulses
        self.update()

    def audio_callback(self, indata, frames, time, status):
        volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
        self.mic_level = min(max(volume_norm * 10, 0.0), 1.0)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        center = self.rect().center()

        for pulse in self.pulses:
            color = QColor(63, 81, 181)
            color.setAlphaF(pulse['opacity'])
            pen = QPen(color)
            pen.setWidth(pulse['width'])
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(center, pulse['radius'], pulse['radius'])

        # Barra de volume no canto inferior direito
        margin_x = 20
        margin_y = 20
        bar_width = 120
        bar_height = 8
        bar_x = self.width() - bar_width - margin_x
        bar_y = self.height() - bar_height - margin_y

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(50, 50, 50))
        painter.drawRoundedRect(bar_x, bar_y, bar_width, bar_height, 4, 4)

        fill_width = int(bar_width * self.mic_level)
        painter.setBrush(QColor(63, 81, 181))
        if fill_width > 0:
            painter.drawRoundedRect(bar_x, bar_y, fill_width, bar_height, 4, 4)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PulseBackground()
    window.show()
    sys.exit(app.exec_())
