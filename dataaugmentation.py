import os
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
from tqdm import tqdm
import random

# Caminhos dos diretórios e arquivos
csv_entrada = "./data/esc50_filtrado.csv"
dir_audios_originais = "./ESC-50-master/audio"
dir_audios_aumentados = "./ESC-50-master/audio_aumentado"

# Criar pasta para salvar os áudios aumentados
os.makedirs(dir_audios_aumentados, exist_ok=True)

# Carregar o CSV filtrado original
df_original = pd.read_csv(csv_entrada)
linhas_novas = []

# Funções de aumento de dados
def aplicar_ruido(y, intensidade=0.002):
    ruido = np.random.randn(len(y))
    return y + intensidade * ruido

def esticar_audio(y, taxa):
    return librosa.effects.time_stretch(y, rate=taxa)

def alterar_tom(y, sr, passos):
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=passos)

def alterar_volume(y, ganho_db):
    return y * (10 ** (ganho_db / 20))

def adicionar_echo(y, sr, atraso=0.1, decaimento=0.3):
    atraso_amostras = int(sr * atraso)
    eco = np.zeros(len(y) + atraso_amostras)
    eco[:len(y)] = y
    eco[atraso_amostras:] += decaimento * y
    eco = eco / np.max(np.abs(eco))  # normalização
    return eco.astype(y.dtype)

def inverter_audio(y):
    return y[::-1]

def normalizar_audio(y):
    maximo = np.max(np.abs(y))
    return y / maximo if maximo > 0 else y

print("Aplicando aumento de dados...")
for _, linha in tqdm(df_original.iterrows(), total=len(df_original)):
    nome_arquivo = linha['filename']
    caminho_completo = os.path.join(dir_audios_originais, nome_arquivo)

    try:
        y, sr = librosa.load(caminho_completo, sr=None)

        # Ruído (30%)
        if random.random() < 0.3:
            y_ruido = aplicar_ruido(y, intensidade=random.uniform(0.001, 0.003))
            nome_ruido = f"noise_{nome_arquivo}"
            sf.write(os.path.join(dir_audios_aumentados, nome_ruido), y_ruido, sr)
            linhas_novas.append({**linha, 'filename': nome_ruido})

        # Alteração de tom (30%)
        if random.random() < 0.3:
            passos = random.choice([-1, 1])
            y_tom = alterar_tom(y, sr, passos=passos)
            y_tom = normalizar_audio(y_tom)
            nome_tom = f"pitch_{nome_arquivo}"
            sf.write(os.path.join(dir_audios_aumentados, nome_tom), y_tom, sr)
            linhas_novas.append({**linha, 'filename': nome_tom})

        # Esticar áudio (30%)
        if random.random() < 0.3:
            taxa = random.choice([0.95, 1.05])
            y_estico = esticar_audio(y, taxa=taxa)
            y_estico = normalizar_audio(y_estico)
            nome_estico = f"stretch_{nome_arquivo}"
            sf.write(os.path.join(dir_audios_aumentados, nome_estico), y_estico, sr)
            linhas_novas.append({**linha, 'filename': nome_estico})

        # Alterar volume (30%)
        if random.random() < 0.3:
            ganho = random.uniform(-3, 3)
            y_volume = alterar_volume(y, ganho)
            y_volume = normalizar_audio(y_volume)
            nome_volume = f"gain_{nome_arquivo}"
            sf.write(os.path.join(dir_audios_aumentados, nome_volume), y_volume, sr)
            linhas_novas.append({**linha, 'filename': nome_volume})

        # Echo (20%)
        if random.random() < 0.2:
            y_eco = adicionar_echo(y, sr, atraso=0.1, decaimento=0.3)
            nome_eco = f"echo_{nome_arquivo}"
            sf.write(os.path.join(dir_audios_aumentados, nome_eco), y_eco, sr)
            linhas_novas.append({**linha, 'filename': nome_eco})

        # Inverter (20%)
        if random.random() < 0.2:
            y_invertido = inverter_audio(y)
            nome_invertido = f"reverse_{nome_arquivo}"
            sf.write(os.path.join(dir_audios_aumentados, nome_invertido), y_invertido, sr)
            linhas_novas.append({**linha, 'filename': nome_invertido})

    except Exception as e:
        print(f"⚠️ Erro ao processar {nome_arquivo}: {e}")

# Concatenar e salvar novo CSV
df_augmentado = pd.DataFrame(linhas_novas)
df_final = pd.concat([df_original, df_augmentado], ignore_index=True)
df_final.to_csv("./data/esc50_aumentado.csv", index=False)

print("✅ Aumentos aplicados e CSV salvo como esc50_aumentado.csv")
