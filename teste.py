import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
from joblib import load
import time


# Parâmetros
DURATION = 5  # segundos de gravação
SR = 44100  # taxa de amostragem usada no treinamento
N_MFCC = 13  # número de MFCCs

# Carregando scaler, label encoder e modelo
scaler = load('./models/scaler.joblib')
le = load('./models/label_encoder.joblib')
model = tf.keras.models.load_model('./models/audio_nn_model.keras')

def record_audio(duration, sr):
    print(f"🎙️ Gravando {duration} segundos de áudio...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    print("✅ Gravação finalizada.")
    return recording.flatten()


def extrair_features(y, sr=44100):
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Delta e Delta-Delta
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    delta_mean = np.mean(delta, axis=1)
    delta2_mean = np.mean(delta2, axis=1)

    # Spectral Features
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # RMS Energy
    rms = np.mean(librosa.feature.rms(y=y))

    # Chroma
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)

    # Concatenar todas as features
    features = np.concatenate([
        mfcc_mean,      # 13
        delta_mean,     # 13
        delta2_mean,    # 13
        [spec_centroid],
        [spec_bw],
        contrast,       # 7
        [rolloff],
        [zcr],
        [rms],
        chroma          # 12
    ])

    return features


def predict_audio():
    audio = record_audio(DURATION, SR)
    features = extrair_features(audio, SR)
    features_scaled = scaler.transform([features])  # aplicar o scaler do treinamento
    prediction = model.predict(features_scaled)
    pred_label = le.inverse_transform([np.argmax(prediction)])
    print(f"🔊 Classe prevista: {pred_label[0]}")



if __name__ == '__main__':
    while True:
        input("\n🔁 Pressione Enter para gravar e classificar o áudio...")
        predict_audio()
        print("\n---")
