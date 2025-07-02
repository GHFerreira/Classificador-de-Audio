import os
import numpy as np
import random
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from joblib import dump
import warnings
import pickle

warnings.filterwarnings("ignore")

# Configurar seeds para reproducibilidade
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Garantir diretório para salvar modelos
os.makedirs("./models", exist_ok=True)

# Carregar dados
X = np.load("./features/X.npy")
y = np.load("./features/y.npy")

# Normalização das features
scaler = StandardScaler()
X = scaler.fit_transform(X)
dump(scaler, "./models/scaler.joblib")

# Encoding das labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
y_cat = to_categorical(y_enc)  # one-hot

# Dividir dados treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.3, random_state=SEED, stratify=y_enc
)

print(f"Treino: {X_train.shape[0]} amostras")
print(f"Teste: {X_test.shape[0]} amostras")

# Construir o modelo sequencial
model = Sequential([
    Dense(256, input_shape=(X.shape[1],), activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_cat.shape[1], activation='softmax')  # número de classes na saída
])

# Compilar o modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping para evitar overfitting
#early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Treinamento
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    #callbacks=[early_stop]
)

# Avaliação
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAcurácia no teste: {acc*100:.2f}%")

# Previsões e relatório de classificação
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

print("\nRelatório de classificação:")
print(classification_report(y_test_labels, y_pred_labels, target_names=le.classes_))

# Salvar modelo, encoder e histórico
model.save("./models/audio_nn_model.keras")
dump(le, "./models/label_encoder.joblib")

with open('./models/history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
