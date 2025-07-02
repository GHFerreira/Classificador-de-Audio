
import os
import pandas as pd

# PATH
CSV_PATH = "./ESC-50-master/meta/esc50.csv"
AUDIO_PATH = "./ESC-50-master/audio"

# Classes
CLASSES_USADAS = [
    'dog', 'cat', 'frog',
    'crying_baby', 'clapping', 'coughing',
    'door_wood_knock', 'keyboard_typing', 'siren', 'clock_alarm', 'church_bells'
]

df = pd.read_csv(CSV_PATH)

# Filtro
df_filtrado = df[df['category'].isin(CLASSES_USADAS)]
print("Total por classe:")
print(df_filtrado['category'].value_counts())

# Novo filtrado
df_filtrado.to_csv("./esc50_filtrado.csv", index=False)
