"""
TREINAR MODELO DENGO - Versão Principal
====================================
Implementações:
1. ✅ Embedding de cidade (modelo sabe QUAL cidade está prevendo)
2. ✅ Attention mechanism para capturar dependências longas
3. ✅ Loss function Huber (mais robusta a outliers que MSE)
4. ✅ Learning rate schedule (decay gradual)
5. ✅ Regularização L2 para evitar overfitting
6. ✅ Ativação ReLU na saída (garante valores não-negativos)
7. ✅ Batch size maior (256) para gradientes mais estáveis
8. ✅ Class weights para balancear cidades grandes/pequenas
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Embedding, Flatten, 
    Concatenate, LayerNormalization, MultiHeadAttention,
    GlobalAveragePooling1D, BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

# Configuração GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduz logs do TensorFlow

print("=" * 60)
print("🚀 TREINAMENTO DENGO - MODELO COM EMBEDDING E ATTENTION")
print("=" * 60)

# --- CONFIGURAÇÕES ---
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 10  # Early stopping
L2_REG = 0.001  # Regularização

# 1. CARREGAR DADOS
print("\n📂 Carregando dados...")

try:
    X_train = np.load('X_train.npy')
    y_train = np.load('y_train.npy')
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    cidades_id_train = np.load('cidades_id_train.npy')
    cidades_id_test = np.load('cidades_id_test.npy')
    
    with open('metadados.json', 'r') as f:
        metadados = json.load(f)
except FileNotFoundError:
    print("❌ Arquivos de dados não encontrados!")
    print("   Execute primeiro: criar_sequencias.py")
    exit()

# Sanitização
X_train = np.nan_to_num(X_train, nan=0.0)
X_test = np.nan_to_num(X_test, nan=0.0)
y_train = np.nan_to_num(y_train, nan=0.0)
y_test = np.nan_to_num(y_test, nan=0.0)

# Dimensões
n_samples_train = X_train.shape[0]
n_samples_test = X_test.shape[0]
n_timesteps = X_train.shape[1]  # Janela temporal (12)
n_features = X_train.shape[2]   # Número de features (15)
n_cidades = metadados['n_cidades']  # ~399

print(f"   Treino: {n_samples_train:,} amostras")
print(f"   Teste:  {n_samples_test:,} amostras")
print(f"   Janela: {n_timesteps} semanas")
print(f"   Features: {n_features}")
print(f"   Cidades: {n_cidades}")

# 2. CALCULAR PESOS DE AMOSTRA (Balancear cidades grandes/pequenas)
print("\n⚖️ Calculando pesos de amostra...")

# Carrega classificação de cidades
try:
    with open('cidades_por_porte.json', 'r', encoding='utf-8') as f:
        cidades_por_porte = json.load(f)
    
    with open('mapeamento_cidades.json', 'r', encoding='utf-8') as f:
        mapeamento = json.load(f)
    
    # Cria mapeamento id -> porte
    id_to_porte = {}
    for porte, cidades in cidades_por_porte.items():
        for cidade in cidades:
            if cidade in mapeamento:
                id_to_porte[mapeamento[cidade]] = porte
    
    # Pesos: cidades grandes têm mais peso (menos comuns no dataset)
    peso_por_porte = {'grande': 3.0, 'media': 2.0, 'pequena': 1.0}
    
    sample_weights = np.array([
        peso_por_porte.get(id_to_porte.get(cid, 'pequena'), 1.0) 
        for cid in cidades_id_train
    ])
    
    print(f"   Peso cidades grandes: 3.0x")
    print(f"   Peso cidades médias:  2.0x")
    print(f"   Peso cidades pequenas: 1.0x")
    
except Exception as e:
    print(f"   ⚠️ Não foi possível calcular pesos: {e}")
    sample_weights = np.ones(n_samples_train)

# 3. CONSTRUIR MODELO COM EMBEDDING E ATTENTION
print("\n🧠 Construindo arquitetura do modelo...")

def build_model_dengo(n_timesteps, n_features, n_cidades, embedding_dim=32):
    """
    Modelo LSTM com:
    - Embedding de cidade
    - Multi-Head Attention
    - Regularização L2
    - Saída não-negativa
    """
    
    # --- INPUT: Série Temporal ---
    serie_input = Input(shape=(n_timesteps, n_features), name='serie_temporal')
    
    # --- INPUT: ID da Cidade ---
    cidade_input = Input(shape=(1,), name='cidade_id')
    
    # --- EMBEDDING DE CIDADE ---
    # Transforma ID numérico em vetor denso de 32 dimensões
    cidade_embed = Embedding(
        input_dim=n_cidades + 1,  # +1 para IDs desconhecidos
        output_dim=embedding_dim,
        name='cidade_embedding'
    )(cidade_input)
    cidade_embed = Flatten()(cidade_embed)
    
    # --- BLOCO LSTM 1 ---
    lstm1 = LSTM(
        units=128, 
        return_sequences=True,
        kernel_regularizer=l2(L2_REG),
        name='lstm_1'
    )(serie_input)
    lstm1 = BatchNormalization()(lstm1)
    lstm1 = Dropout(0.2)(lstm1)
    
    # --- MULTI-HEAD ATTENTION ---
    # Permite ao modelo "focar" em semanas mais importantes
    attention = MultiHeadAttention(
        num_heads=4,
        key_dim=32,
        name='attention'
    )(lstm1, lstm1)
    attention = LayerNormalization()(attention + lstm1)  # Residual connection
    
    # --- BLOCO LSTM 2 ---
    lstm2 = LSTM(
        units=64, 
        return_sequences=False,
        kernel_regularizer=l2(L2_REG),
        name='lstm_2'
    )(attention)
    lstm2 = BatchNormalization()(lstm2)
    lstm2 = Dropout(0.2)(lstm2)
    
    # --- CONCATENAR LSTM + EMBEDDING ---
    merged = Concatenate(name='merge')([lstm2, cidade_embed])
    
    # --- CAMADAS DENSAS ---
    dense1 = Dense(64, activation='relu', kernel_regularizer=l2(L2_REG))(merged)
    dense1 = Dropout(0.1)(dense1)
    
    dense2 = Dense(32, activation='relu', kernel_regularizer=l2(L2_REG))(dense1)
    
    # --- SAÍDA ---
    # Usamos 'relu' para garantir que a previsão nunca seja negativa
    output = Dense(1, activation='relu', name='output')(dense2)
    
    # --- COMPILAR MODELO ---
    model = Model(
        inputs=[serie_input, cidade_input],
        outputs=output,
        name='DengoAI'
    )
    
    return model

model = build_model_dengo(n_timesteps, n_features, n_cidades)

# --- COMPILAÇÃO ---
# Huber Loss: Mais robusta a outliers que MSE
optimizer = Adam(learning_rate=LEARNING_RATE)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.Huber(delta=1.0),  # Huber é menos sensível a outliers
    metrics=['mae', 'mse']
)

print("\n📋 Resumo da Arquitetura:")
model.summary()

# 4. CALLBACKS
print("\n⚙️ Configurando callbacks...")

callbacks = [
    # Para se não melhorar
    EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Salva o melhor modelo
    ModelCheckpoint(
        'dengo_ai.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    
    # Reduz learning rate se platô
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Reduz LR pela metade
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# 5. TREINAR!
print("\n" + "=" * 60)
print("🔥 INICIANDO TREINAMENTO")
print("=" * 60)

history = model.fit(
    [X_train, cidades_id_train],  # 2 inputs
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=([X_test, cidades_id_test], y_test),
    callbacks=callbacks,
    sample_weight=sample_weights,
    verbose=1
)

# 6. AVALIAÇÃO RÁPIDA
print("\n📊 Avaliação no conjunto de teste...")
results = model.evaluate([X_test, cidades_id_test], y_test, verbose=0)
print(f"   Loss (Huber): {results[0]:.6f}")
print(f"   MAE: {results[1]:.6f}")
print(f"   MSE: {results[2]:.6f}")

# Calcular R² manualmente
y_pred = model.predict([X_test, cidades_id_test], verbose=0).flatten()
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - (ss_res / ss_tot)
print(f"   R²: {r2:.4f}")

# 7. GRÁFICOS
print("\n📈 Gerando gráficos...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Gráfico 1: Loss
axes[0, 0].plot(history.history['loss'], label='Treino')
axes[0, 0].plot(history.history['val_loss'], label='Validação')
axes[0, 0].set_title('Curva de Aprendizado (Loss)')
axes[0, 0].set_xlabel('Épocas')
axes[0, 0].set_ylabel('Huber Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Gráfico 2: MAE
axes[0, 1].plot(history.history['mae'], label='Treino')
axes[0, 1].plot(history.history['val_mae'], label='Validação')
axes[0, 1].set_title('Erro Absoluto Médio (MAE)')
axes[0, 1].set_xlabel('Épocas')
axes[0, 1].set_ylabel('MAE')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Gráfico 3: Previsão vs Real (scatter)
sample_idx = np.random.choice(len(y_test), min(5000, len(y_test)), replace=False)
axes[1, 0].scatter(y_test[sample_idx], y_pred[sample_idx], alpha=0.3, s=1)
axes[1, 0].plot([0, y_test.max()], [0, y_test.max()], 'r--', label='Ideal')
axes[1, 0].set_title(f'Previsão vs Real (R² = {r2:.4f})')
axes[1, 0].set_xlabel('Valor Real (normalizado)')
axes[1, 0].set_ylabel('Valor Previsto (normalizado)')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Gráfico 4: Distribuição dos erros
erros = y_test - y_pred
axes[1, 1].hist(erros, bins=100, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--', label='Erro zero')
axes[1, 1].set_title(f'Distribuição dos Erros (μ={np.mean(erros):.4f}, σ={np.std(erros):.4f})')
axes[1, 1].set_xlabel('Erro (Real - Previsto)')
axes[1, 1].set_ylabel('Frequência')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('grafico_treinamento.png', dpi=150)
print("   Gráfico salvo: grafico_treinamento.png")

# 8. SALVAR HISTÓRICO
historico = {
    'loss': [float(x) for x in history.history['loss']],
    'val_loss': [float(x) for x in history.history['val_loss']],
    'mae': [float(x) for x in history.history['mae']],
    'val_mae': [float(x) for x in history.history['val_mae']],
    'r2_final': float(r2),
    'melhor_epoca': int(np.argmin(history.history['val_loss']) + 1)
}

with open('historico_treinamento.json', 'w') as f:
    json.dump(historico, f, indent=2)

print("\n" + "=" * 60)
print("✅ TREINAMENTO CONCLUÍDO!")
print("=" * 60)
print(f"\n📋 RESULTADOS:")
print(f"   • R² Final: {r2:.4f}")
print(f"   • MAE Final: {results[1]:.6f}")
print(f"   • Melhor época: {historico['melhor_epoca']}")
print(f"\n📁 Arquivos salvos:")
print(f"   • dengo_ai.keras (modelo)")
print(f"   • grafico_treinamento.png")
print(f"   • historico_treinamento.json")

# 9. COMPARAÇÃO COM BASELINE
print("\n📊 Comparação com Baseline...")

# Baseline: prever o último valor da janela (semana anterior)
baseline_pred = X_test[:, -1, 0]  # Última semana, primeira feature (casos_est)
baseline_ss_res = np.sum((y_test - baseline_pred) ** 2)
baseline_r2 = 1 - (baseline_ss_res / ss_tot)

print(f"   R² Modelo Dengo: {r2:.4f}")
print(f"   R² Baseline (semana anterior): {baseline_r2:.4f}")
print(f"   Melhoria: {(r2 - baseline_r2) * 100:.1f}%")

if r2 > baseline_r2:
    print("   ✅ Modelo Dengo é MELHOR que o baseline!")
else:
    print("   ⚠️ Modelo Dengo precisa de mais ajustes.")