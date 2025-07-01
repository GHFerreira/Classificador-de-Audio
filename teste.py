import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
from joblib import load
import time

# Parâmetros
DURATION = 5  # segundos de gravação
SR = 44100  # taxa de amostragem usada no treinamento
N_MFCC = 13 # numero de características extraídas

# Carregando scaler, label encoder e modelo
scaler = load('./models/scaler.joblib')
le = load('./models/label_encoder.joblib')
model = tf.keras.models.load_model('./models/audio_nn_model.keras')

def record_audio(duration, sr):
    print(f"Gravando {duration} segundos de áudio...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    print("Gravação finalizada.")
    return recording.flatten()

def extract_mfcc(audio, sr, n_mfcc):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

def predict_audio():
    audio = record_audio(DURATION, SR)
    mfcc_features = extract_mfcc(audio, SR, N_MFCC)
    mfcc_scaled = scaler.transform([mfcc_features])  # aplicar scaler do treino
    prediction = model.predict(mfcc_scaled)
    pred_label = le.inverse_transform([np.argmax(prediction)])
    print(f"Classe prevista: {pred_label[0]}")

if __name__ == '__main__':
    while True:
        input("Pressione Enter para gravar um áudio e classificar...")
        predict_audio()
        print("\n---\n")
