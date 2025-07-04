import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

import os
import shutil

orig = './ESC-50-master/audio'
dest = './ESC-50-master/audio_aumentado'

for f in os.listdir(orig):
    if f.endswith('.wav'):
        shutil.copy(os.path.join(orig, f), os.path.join(dest, f))

# Parâmetros
CSV_AUMENTADO = "./data/esc50_aumentado.csv"
AUDIO_PATH = "./ESC-50-master/audio_aumentado"  # contém originais e aumentados
N_MFCC = 13

# Carregar csv
df = pd.read_csv(CSV_AUMENTADO)

X = []
y = []

def extrair_features(caminho_audio):
    y, sr = librosa.load(caminho_audio, sr=44100)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Delta e Delta-Delta
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    delta_mean = np.mean(delta, axis=1)
    delta2_mean = np.mean(delta2, axis=1)

    # Spectral
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # RMS Energy
    rms = np.mean(librosa.feature.rms(y=y))

    # Chroma
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)

    # Concatenar todas as features
    features = np.concatenate([
        mfcc_mean,       # 13 coeficientes MFCC
        delta_mean,      # 13 coeficientes Delta (1ª derivada dos MFCC)
        delta2_mean,     # 13 coeficientes Delta-Delta (2ª derivada dos MFCC)
        [spec_centroid], # 1 valor: centroide espectral
        [spec_bw],       # 1 valor: largura de banda espectral
        contrast,        # 7 valores: contraste espectral
        [rolloff],       # 1 valor: roll-off espectral
        [zcr],           # 1 valor: taxa de cruzamento por zero
        [rms],           # 1 valor: energia RMS
        chroma           # 12 valores: intensidade das notas musicais (chroma)
        # Total: 13 + 13 + 13 + 1 + 1 + 7 + 1 + 1 + 1 + 12 = 63
    ])

    return features

print("Extraindo features dos áudios...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    caminho = os.path.join(AUDIO_PATH, row['filename'])
    try:
        vetor_mfcc = extrair_features(caminho)
        X.append(vetor_mfcc)
        y.append(row['category'])
    except Exception as e:
        print(f"⚠️ Erro ao processar {row['filename']}: {e}")

X = np.array(X)
y = np.array(y)

# Salvar
os.makedirs("features", exist_ok=True)
np.save("./features/X.npy", X)
np.save("./features/y.npy", y)

print(f"\n✅ Extração concluída! {X.shape[0]} amostras e {X.shape[1]} features por amostra.")