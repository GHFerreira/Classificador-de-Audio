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

# Criar pasta se não existir
os.makedirs(augmented_path, exist_ok=True)

# Carregar CSV original
df = pd.read_csv(csv_original)
new_rows = []

# Funções de aumentos
def add_noise(y, noise_level=0.002):
    noise = np.random.randn(len(y))
    return y + noise_level * noise

def stretch_audio(y, rate):
    return librosa.effects.time_stretch(y, rate=rate)

def shift_pitch(y, sr, n_steps):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

def change_gain(y, gain_db):
    return y * (10 ** (gain_db / 20))

def add_echo(y, sr, delay=0.1, decay=0.3):
    n_delay = int(sr * delay)
    echo = np.zeros(len(y) + n_delay)
    echo[:len(y)] = y
    echo[n_delay:] += decay * y
    # Normalize to prevent clipping
    echo = echo / np.max(np.abs(echo))
    return echo.astype(y.dtype)

def reverse_audio(y):
    return y[::-1]

def normalize_audio(y):
    max_abs = np.max(np.abs(y))
    if max_abs > 0:
        y = y / max_abs
    return y

print(" Aplicando aumento de dados...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    filename = row['filename']
    path = os.path.join(audio_path, filename)

    try:
        y, sr = librosa.load(path, sr=None)

        # Noise (30% chance)
        if random.random() < 0.3:
            y_noise = add_noise(y, noise_level=random.uniform(0.001, 0.003))
            fn_noise = f"noise_{filename}"
            sf.write(os.path.join(augmented_path, fn_noise), y_noise, sr)
            new_rows.append({**row, 'filename': fn_noise})

        # Pitch shift (30% chance)
        if random.random() < 0.3:
            steps = random.choice([-1, 1])
            y_pitch = shift_pitch(y, sr, n_steps=steps)
            y_pitch = normalize_audio(y_pitch)
            fn_pitch = f"pitch_{filename}"
            sf.write(os.path.join(augmented_path, fn_pitch), y_pitch, sr)
            new_rows.append({**row, 'filename': fn_pitch})

        # Time stretch (30% chance)
        if random.random() < 0.3:
            rate = random.choice([0.95, 1.05])
            y_stretch = stretch_audio(y, rate=rate)
            y_stretch = normalize_audio(y_stretch)
            fn_stretch = f"stretch_{filename}"
            sf.write(os.path.join(augmented_path, fn_stretch), y_stretch, sr)
            new_rows.append({**row, 'filename': fn_stretch})

        # Gain (volume) change (30% chance)
        if random.random() < 0.3:
            gain_db = random.uniform(-3, 3)  # -3dB a +3dB
            y_gain = change_gain(y, gain_db)
            y_gain = normalize_audio(y_gain)
            fn_gain = f"gain_{filename}"
            sf.write(os.path.join(augmented_path, fn_gain), y_gain, sr)
            new_rows.append({**row, 'filename': fn_gain})

        # Echo (20% chance)
        if random.random() < 0.2:
            y_echo = add_echo(y, sr, delay=0.1, decay=0.3)
            fn_echo = f"echo_{filename}"
            sf.write(os.path.join(augmented_path, fn_echo), y_echo, sr)
            new_rows.append({**row, 'filename': fn_echo})

        # Reverse (20% chance)
        if random.random() < 0.2:
            y_rev = reverse_audio(y)
            fn_rev = f"reverse_{filename}"
            sf.write(os.path.join(augmented_path, fn_rev), y_rev, sr)
            new_rows.append({**row, 'filename': fn_rev})

    except Exception as e:
        print(f"⚠️ Erro ao processar {filename}: {e}")

# Novo dataframe com os aumentos
df_augmented = pd.DataFrame(new_rows)
df_total = pd.concat([df, df_augmented], ignore_index=True)

# Salvar CSV atualizado
df_total.to_csv("./esc50_aumentado_calibrado.csv", index=False)
print("✅ Aumentos aplicados e CSV salvo como esc50_aumentado_calibrado.csv")
