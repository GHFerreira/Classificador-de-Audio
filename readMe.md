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

As classes selecionadas são divididas em 3 grupos:
**Animais: Cachorro, gato e sapo.
Sons humanos: choro de bebê, palmas e tosse.
sons cotidianos: batida na porta, teclado, sirene, alarme e sino.**

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

Esse filtro tem como saída um novo arquvio csv **(.\data\esc50_filtrado.csv)** de mesma estrutura que o anterior. 

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