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
dest = './ESC-50-master/audio_augmented'

for f in os.listdir(orig):
    if f.endswith('.wav'):
        shutil.copy(os.path.join(orig, f), os.path.join(dest, f))

# Parâmetros
CSV_AUMENTADO = "./esc50_aumentado.csv"
AUDIO_PATH = "./ESC-50-master/audio_augmented"  # contém originais e aumentados
N_MFCC = 13

# Carregar csv
df = pd.read_csv(CSV_AUMENTADO)

X = []
y = []

def extrair_mfcc(caminho_audio):
    y, sr = librosa.load(caminho_audio, sr=44100)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

print("Extraindo MFCCs dos áudios aumentados...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    caminho = os.path.join(AUDIO_PATH, row['filename'])
    try:
        vetor_mfcc = extrair_mfcc(caminho)
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
