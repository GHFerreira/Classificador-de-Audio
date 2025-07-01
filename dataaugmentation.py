import os
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm
import random

# Caminhos
csv_original = "./esc50_filtrado.csv"
audio_path = "./ESC-50-master/audio"
augmented_path = "./ESC-50-master/audio_augmented"

# Cria pasta se não existir
os.makedirs(augmented_path, exist_ok=True)

# Carregamento
df = pd.read_csv(csv_original)
new_rows = []

# Funções de aumento calibradas
def add_noise(y, noise_level=0.003):  # mais sutil
    noise = np.random.randn(len(y))
    return y + noise_level * noise

def stretch_audio(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)

def shift_pitch(y, sr, n_steps):
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

print("🎧 Aplicando aumentos de dados calibrados...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    filename = row['filename']
    path = os.path.join(audio_path, filename)

    try:
        y, sr = librosa.load(path, sr=None)

        # Noise (com 50% de chance)
        if random.random() < 0.5:
            y_noise = add_noise(y, noise_level=random.uniform(0.001, 0.004))
            fn_noise = f"noise_{filename}"
            sf.write(os.path.join(augmented_path, fn_noise), y_noise, sr)
            new_rows.append({**row, 'filename': fn_noise})

        # Pitch (com 50% de chance)
        if random.random() < 0.5:
            steps = random.choice([-1, 1])
            y_pitch = shift_pitch(y, sr, n_steps=steps)
            fn_pitch = f"pitch_{filename}"
            sf.write(os.path.join(augmented_path, fn_pitch), y_pitch, sr)
            new_rows.append({**row, 'filename': fn_pitch})

        # Time stretch (com 50% de chance)
        if random.random() < 0.5:
            rate = random.choice([0.95, 1.05])
            y_stretch = stretch_audio(y, rate=rate)
            fn_stretch = f"stretch_{filename}"
            sf.write(os.path.join(augmented_path, fn_stretch), y_stretch, sr)
            new_rows.append({**row, 'filename': fn_stretch})

    except Exception as e:
        print(f"⚠️ Erro ao processar {filename}: {e}")

# Novo dataframe com os aumentos
df_augmented = pd.DataFrame(new_rows)
df_total = pd.concat([df, df_augmented], ignore_index=True)

# Salva o novo CSV
df_total.to_csv("./esc50_aumentado.csv", index=False)
print("✅ Aumentos aplicados e CSV salvo como esc50_aumentado.csv")
