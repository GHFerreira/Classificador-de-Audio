# 🔈 Classificador de Áudio - Uma aplicação teórica de Sinais e Sistemas Lineares com Ciência de Dados

## 📖 Descrição Geral

Este projeto tem como objetivo criar um sistema de classificação automática de sons usando gravações de áudio. A base de dados utilizada é o **ESC-50**, que reúne amostras sonoras de diversas categorias, como sons de animais, alarmes, instrumentos e ambientes.

Para as 11 categorias escolhidas, foi aplicada a técnica de **data augmentation** (aumento de dados), gerando variações artificiais dos áudios originais e ampliando a quantidade de exemplos disponíveis para o treinamento. Em seguida, são extraídas **63 features** (características) de cada amostra, como MFCCs, centroides espectrais, contraste, entre outras.

Essas características alimentam um modelo de rede neural treinado para reconhecer os padrões sonoros. O sistema é integrado a uma interface gráfica intuitiva, que permite gravar áudio ao vivo e exibir, de forma prática, a classe sonora prevista com base no modelo treinado.


## ⚙ Requisitos

O projeto foi desenvolvido em **Python 3.12.7** e depende das seguintes bibliotecas:

- `numpy`
- `pandas`
- `librosa`
- `scikit-learn`
- `sounddevice`
- `matplotlib`
- `joblib`
- `streamlit`
- `tensorflow`
- `tqdm`
- `seaborn`
- `PyQt5`

Todas as dependências podem ser instaladas com o seguinte comando:

```bash
pip install -r requirements.txt
```

Não há requisitos específicos de hardware e pode ser executado em qualquer máquina com suporte para Python e as bibliotecas acima.

## 🏗 Estrutura do código

De forma geral o projeto está centrado em 5 arquivos .py, sendo 4 de configuração (**selecionarclasses.py**, **dataaugmentation.py**, **features.py** e **classificador.py**) e 1 de execução do app (**app.py**). 

###### esc50.csv (.\ESC-50-master\meta\esc50.csv)

Esse arquivo csv contém, dentre outras features, as informações de classe e nome do arquivo de áudio em questão. É ele quem torna possível a navegação pelos arquivos de áudio. Cada áudio tem 5 segundos de duração.

###### Arquivo 1 - selecionarclasses.py
Esse arquivo é o ponto de partida do projeto, nele, está o script responsável por filtrar as classes e arquivos de áudio selecionadas para o processo de treinamento do modelo.

As classes são selecionadas a partir de um vetor de strings contendo o nome de cada uma, e depois, um filtro é aplicado no arquivo esc50.csv para que apenas os arquivos correspondentes possam ser acessados.

As classes selecionadas são divididas em 3 grupos:<br>
- **Animais: Cachorro, gato e sapo**
- **Sons humanos: choro de bebê, palmas e tosse**
- **Sons cotidianos: batida na porta, teclado, sirene, alarme e sino.**

Esses serão os sons que o app será capaz de reconhecer.

```bash
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
```

Esse filtro tem como saída um novo arquivo csv **(.\data\esc50_filtrado.csv)** de mesma estrutura que o anterior. 

Além disso, é mostrado no terminal a quantidade de arquivos de áudio por classe, totalizando 40 arquivos por classe, totalizando 440 arquivos de entrada inicial.
<p align="center">
  <img src="imgs\classes.png" alt="Arquivos por classe" width="200">
</p>
 
###### Arquivo 2 - dataaugmentation.py
Foi constatado que, com apenas os 440 arquivos de áudio originais o treinamento não tinha dados o suficiente para ter uma boa eficácia nas classificações.
Para resolver isso, foi necessário aplicar efeitos como: **ruído**, **esticar audio**, **alteração de tom**, **variação de volume**, **echo** e **inversão** aos arquivos originais para que novas variações de cada um sejam criadas.

As funções que descrevem a aplicação desses efeitos podem ser vistas abaixo:
```bash
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
    eco = eco / np.max(np.abs(eco)) 
    return eco.astype(y.dtype)

def inverter_audio(y):
    return y[::-1]
```
A aplicação desses efeitos para cada áudio é feita de forma randômica, com cada áudio tendo 30% de chance de ter  **ruído**, **esticar audio**, **alteração de tom** e **variação de volume** e 20% de chance de ter **echo** e **inversão** aplicados.
É importante que seja feito dessa forma randômica, pois, a cada execução, os novos áudios gerados com os efeitos aplicados são diferentes. Isso previne um superajuste nos dados e o enviesamento do modelo.

```bash
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
```

Após esse processo, o número de arquivos salta de **400** para **1176**, trazendo mais robusutez à massa de dados e, consequentemente, mais qualidade ao treinamento do modelo.
Como saída, esse processo retorna um novo arquivo csv **(.\data\esc50_aumentado.csv)** com os mesmos moldes dos anteriores, porém, contendo agora também os arquivos gerados com efeitos aplicados.

###### Arquivo 3 - features.py
Esse arquivo é responsável por fazer a extração das features dos arquivos de áudio.
A extração de características foi realizada transformando os sinais de áudio em representações numéricas que pudessem ser utilizadas pelo classificador. Esse processo foi aplicado tanto às amostras originais quanto às geradas via data augmentation, totalizando 1176 arquivos processados.

O procedimento consistiu nas seguintes etapas:

1) **Leitura dos arquivos de áudio:** Cada amostra foi carregada com taxa de amostragem de 44.100 Hz.

```bash
y, sr = librosa.load(caminho_audio, sr=44100)
```

2) **Extração de 13 MFCC's (Mel-Frequency Cepstral Coefficients):**
Representam a forma do espectro sonoro e são amplamente utilizados em reconhecimento de fala e sons ambientais.

```bash
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
```

3) **Delta e Delta-Delta (13 + 13 coeficientes):**
Primeira e segunda derivadas dos MFCCs, que capturam a variação e aceleração das mudanças temporais.

```bash
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    delta_mean = np.mean(delta, axis=1)
    delta2_mean = np.mean(delta2, axis=1)
```

4) **7 Características espectrais:**
- **Spectral Centroid (centro de massa espectral)**
```bash
spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
```

- **Spectral Bandwidth (largura de banda)**
```bash
spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
```

- **Spectral Contrast (contraste entre bandas)**
```bash
contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
```

- **Spectral Roll-off (frequência limite de energia acumulada)**
```bash
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
```

5) **ZCR (Zero Crossing Rate):** Quantidade de vezes que o sinal cruza o zero — útil para identificar sons ruidosos ou tonais

```bash
zcr = np.mean(librosa.feature.zero_crossing_rate(y))
```

6) **RMS (Root Mean Square Energy):** Energia média do sinal, relacionada ao volume.

```bash
rms = np.mean(librosa.feature.rms(y=y))
```

7) **Chroma STFT (12 coeficientes):** Intensidade das 12 notas musicais
```bash
chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
```

8) Por fim, todas as features são concatenadas em um único vetor, representando todas as características extraídas do arquivo de áudio

```bash
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
```
**Porque calcular a média?**

Como a maioria das características são inicialmente matrizes com variações ao longo do tempo, é necessário converter esses dados para um vetor de dimensão fixa.
 Isso é feito por meio do cálculo da média de cada coeficiente ao longo do tempo, gerando uma representação compacta da tendência geral do áudio, viabilizando seu uso em classificadores.

Ao fim do processo dois vetores são salvos, representando as features extraídas dos arquivos de áudio:
- **X (./features/X.npy):** Vetores com as 63 características extraídas

- **y (./features/y.npy):** Classes correspondentes

###### Arquivo 4 - Classificador.py
Esse arquivo contém a construção do modelo classificador.
O processo de classificação foi realizado utilizando uma rede neural densa (fully-connected) construída com a biblioteca TensorFlow/Keras.

Os vetores citados anteriormente, são lidos como entrada para o treinamento.

Antes de serem submentidos ao treinamento, é necessário um tratamento nessas features.

Para o vetor **X**, que contém os valores numéricos, é necessário uma normalização dos dados, garantindo que todas as features tenham média 0 e desvio padrão 1. Isso acelera e estabiliza o processo de aprendizado do modelo. O scaler é salvo para uso posterior no tratamento do áudio a ser previsto

```bash
scaler = StandardScaler()
X = scaler.fit_transform(X)
dump(scaler, "./models/scaler.joblib")
```

Para o vetor **y**, que contém as "etiquetas" de cada feature, é usado LabelEncoder para codificar numericamente essas classes.
Depois, é usado one-hot enconding para representar vetroialmente essa classe dentro do conjunto.
Isso é necessário para que a função de saída 'Softmax' na rede seja usada.

```bash
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)  # one-hot encoding
```

Além disso, os dados são dividos em arquivos de treino e teste, usando a função **train_test_split** do Scikitlearn, ficando:
- 823 para treino **(70%)**
- 353 para teste **(30%)**

**Arquitetura da rede:**
O modelo possui a seguinte estrutura:
```
Camada de entrada com 63 neurônios (uma para cada feature extraída).

Camada densa com 256 unidades e ativação ReLU + Dropout (0.4).

Camada densa com 128 unidades e ativação ReLU + Dropout (0.3).

Camada densa com 64 unidades e ativação ReLU + Dropout (0.2).

Camada de saída com número de neurônios igual ao total de classes, com ativação softmax.
```
Essa arquitetura foi escolhida por ser leve, eficiente e suficientemente expressiva para o volume de dados trabalhado.

O treinamento foi realizado com **100 Epochs** (vezes que os dados são submetidos à rede) e tem acurácia em torno de **89%**.

<p align="center">
  <img src="imgs\treinamento.png" alt="Arquivos por classe" width="400">
</p>

Posteriormente o modelo, o encoder e o histórico de treinamento são salvos para uso futuro.

```bash
    model.save("./models/audio_nn_model.keras")
    dump(le, "./models/label_encoder.joblib")

    with open('./models/history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
```

###### Arquivo 5 - app.py
Esse arquivo contém a interface gráfica interativa para capturar áudio do microfone, extrair suas features acústicas e classificá-lo com base no modelo treinado.

A interface segue um modelo simples, com o seguinte fluxo de interação:
1) O usuário clica no botão.

2) A interface grava o áudio.

3) A gravação termina e é processada.

4) A classe prevista é exibida na tela.

A interface tem 3 telas correspondentes e podem ser vistas abaixo:

1) A estrutura do app é simples, composta apenas por um botão e uma barra que representa o volume, presente no canto inferior direito.

<p align="center">
  <img src="imgs\tela1.png" alt="" width="400">
</p><br>

2) Ao clicar no botão, uma animação é iniciada, e uma mensagem de "gravando" é exibida, para mostrar ao usuário que a captação de áudio está ativa por 5 segundos. O nível do microfone pode ser monitorado pela barra já citada.

<p align="center">
  <img src="imgs\tela2.png" alt="" width="400">
</p><br>

3) Ao fim do processamento, a classe é exibida ao usuário, e o botão fica disponível para uma nova classificação, voltando ao estado inicial.

<p align="center">
  <img src="imgs\tela3.png" alt="" width="400">
</p><br>

## 🔎 Testes

De forma geral, o desempenho do modelo durante os testes se mostrou satisfatório ao prever corretamente as classes com os quais foi treinado.

É importante ressaltar que a qualidade da captação tem impacto direto no funcionamento do app e na qualidade dos resultados, portanto, é recomendado que a captação seja feita usando equipamentos iguais ou semelhantes aos listados abaixos:

**Especificações do equipamento usado para teste:**

| Componente       | Especificação                                       |
|------------------|-----------------------------------------------------|
| **Processador**  | AMD A10-9700 RADEON R7, 10 Compute Cores (4C + 6G), 3.50 GHz |
| **RAM instalada**| 16,0 GB                        |
| **Placa de vídeo**| AMD Radeon R7 Graphics (998 MB)                    |
| **Armazenamento**| 224 GB SSD SATA                                     |
| **Microfone**| HyperX QuadCast2                     |