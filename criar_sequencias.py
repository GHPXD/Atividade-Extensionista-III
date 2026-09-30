"""
CRIAR SEQUÊNCIAS DENGO - Versão Principal
======================================
Implementações:
1. ✅ Normalização POR CIDADE (RobustScaler)
2. ✅ Janela aumentada para 12 semanas (3 meses)
3. ✅ Features derivadas (lag, diff, média móvel)
4. ✅ Features de sazonalidade (seno/cosseno)
5. ✅ Encoding de cidade para embedding
6. ✅ Separação por porte de cidade
7. ✅ Validação cruzada temporal (K-Fold)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, LabelEncoder
import joblib
import json
from pathlib import Path

# --- CONFIGURAÇÕES ---
ARQUIVO_ENTRADA = "DATASET_PARA_IA.csv"
TAMANHO_JANELA = 12  # 12 semanas = ~3 meses (ciclo epidemiológico)
DATA_CORTE = np.datetime64('2025-01-01')

# Features originais (SEM Rt - data leakage removido!)
COLUNAS_CLIMA = [
    'tempmed', 'tempmin', 'tempmax',
    'umidmed', 'umidmin', 'umidmax',
    'receptivo'
]

COLUNA_ALVO = 'casos_est'

print("=" * 60)
print("🚀 CRIAÇÃO DE SEQUÊNCIAS DENGO - VERSÃO PRINCIPAL")
print("=" * 60)

# 1. CARREGAR DATASET
print("\n📂 Carregando dataset...")
df = pd.read_csv(ARQUIVO_ENTRADA)
df['data'] = pd.to_datetime(df['data'])
df.sort_values(by=['cidade', 'data'], inplace=True)

print(f"   Linhas: {len(df):,}")
print(f"   Cidades: {df['cidade'].nunique()}")

# 2. ENGENHARIA DE FEATURES
print("\n🧠 Criando features derivadas...")

# A. Features de sazonalidade (ciclos anuais)
df['semana_ano'] = df['SE'] % 100
df['seno_tempo'] = np.sin(2 * np.pi * df['semana_ano'] / 52.1775)
df['cosseno_tempo'] = np.cos(2 * np.pi * df['semana_ano'] / 52.1775)

# B. Features derivadas por cidade (lag, diff, média móvel)
print("   Calculando lags e médias móveis (por cidade)...")

# Agrupa por cidade para calcular features temporais
df = df.sort_values(['cidade', 'data'])

# Lag features (semanas anteriores)
for lag in [1, 2, 4]:
    df[f'casos_lag{lag}'] = df.groupby('cidade')[COLUNA_ALVO].shift(lag)

# Diferença (taxa de variação)
df['casos_diff'] = df.groupby('cidade')[COLUNA_ALVO].diff()

# Médias móveis
df['casos_ma4'] = df.groupby('cidade')[COLUNA_ALVO].transform(
    lambda x: x.rolling(window=4, min_periods=1).mean()
)
df['casos_ma8'] = df.groupby('cidade')[COLUNA_ALVO].transform(
    lambda x: x.rolling(window=8, min_periods=1).mean()
)

# Preencher NaNs criados pelos lags (primeiras semanas de cada cidade)
cols_derivadas = ['casos_lag1', 'casos_lag2', 'casos_lag4', 'casos_diff', 'casos_ma4', 'casos_ma8']
for col in cols_derivadas:
    df[col] = df.groupby('cidade')[col].ffill().bfill()
    # Se ainda tiver NaN, preenche com 0
    df[col] = df[col].fillna(0)

# C. Encoding de cidade para embedding
print("   Criando encoding de cidades...")
label_encoder = LabelEncoder()
df['cidade_id'] = label_encoder.fit_transform(df['cidade'])
joblib.dump(label_encoder, 'label_encoder_cidades.pkl')

# Salvar mapeamento cidade -> id
mapeamento_cidades = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
with open('mapeamento_cidades.json', 'w', encoding='utf-8') as f:
    json.dump(mapeamento_cidades, f, ensure_ascii=False, indent=2)

# Lista final de features
COLUNAS_FEATURES = [
    # Casos (alvo e derivados)
    'casos_est', 'casos_lag1', 'casos_lag2', 'casos_lag4',
    'casos_diff', 'casos_ma4', 'casos_ma8',
    # Clima
    'tempmed', 'tempmin', 'tempmax',
    'umidmed', 'umidmin', 'umidmax',
    'receptivo',
    # Sazonalidade
    'seno_tempo', 'cosseno_tempo'
]

print(f"   Total de features: {len(COLUNAS_FEATURES)}")
print(f"   Features: {COLUNAS_FEATURES}")

# 3. NORMALIZAÇÃO POR CIDADE (RobustScaler)
print("\n📏 Normalizando dados POR CIDADE (RobustScaler)...")

df_scaled = df.copy()
scalers_por_cidade = {}

# Normaliza cada cidade separadamente
for cidade in df['cidade'].unique():
    mask = df['cidade'] == cidade
    
    # RobustScaler é mais robusto a outliers que MinMaxScaler
    scaler = RobustScaler()
    df_scaled.loc[mask, COLUNAS_FEATURES] = scaler.fit_transform(
        df.loc[mask, COLUNAS_FEATURES]
    )
    scalers_por_cidade[cidade] = scaler

# Salvar scalers para uso em produção
joblib.dump(scalers_por_cidade, 'scalers_por_cidade.pkl')
print(f"   ✅ {len(scalers_por_cidade)} scalers salvos (um por cidade)")

# 4. CRIAR SEQUÊNCIAS (JANELAS DESLIZANTES)
print(f"\n🔄 Criando janelas de {TAMANHO_JANELA} semanas...")

X = []          # Features (janela de entrada)
y = []          # Alvo (próxima semana)
cidades = []    # Nome da cidade
cidades_id = [] # ID numérico da cidade (para embedding)
datas = []      # Data do alvo

dados_por_cidade = df_scaled.groupby('cidade')
total_cidades = len(dados_por_cidade)
contador = 0

for nome_cidade, grupo in dados_por_cidade:
    valores = grupo[COLUNAS_FEATURES].values
    target = grupo[COLUNA_ALVO].values  # Já normalizado
    cidade_id = grupo['cidade_id'].values[0]
    datas_grupo = grupo['data'].values
    
    # Só cria sequências se a cidade tem dados suficientes
    if len(grupo) > TAMANHO_JANELA:
        for i in range(len(grupo) - TAMANHO_JANELA):
            X.append(valores[i : i + TAMANHO_JANELA])
            y.append(target[i + TAMANHO_JANELA])
            cidades.append(nome_cidade)
            cidades_id.append(cidade_id)
            datas.append(datas_grupo[i + TAMANHO_JANELA])
    
    contador += 1
    if contador % 100 == 0:
        print(f"   Processado: {contador}/{total_cidades} cidades")

# Converter para arrays numpy
X = np.array(X)
y = np.array(y)
cidades = np.array(cidades)
cidades_id = np.array(cidades_id)
datas = np.array(datas)

print(f"\n📊 Formato dos dados:")
print(f"   X (entrada):    {X.shape}  (Amostras, Janela, Features)")
print(f"   y (alvo):       {y.shape}  (Amostras,)")
print(f"   cidades_id:     {cidades_id.shape}  (Para embedding)")

# 5. DIVISÃO TREINO/TESTE (TEMPORAL)
print("\n✂️ Separando Treino (2015-2024) e Teste (2025)...")

idx_treino = datas < DATA_CORTE
idx_teste = datas >= DATA_CORTE

X_train, X_test = X[idx_treino], X[idx_teste]
y_train, y_test = y[idx_treino], y[idx_teste]
cidades_train, cidades_test = cidades[idx_treino], cidades[idx_teste]
cidades_id_train, cidades_id_test = cidades_id[idx_treino], cidades_id[idx_teste]
datas_train, datas_test = datas[idx_treino], datas[idx_teste]

print(f"   Treino (2015-2024): {X_train.shape[0]:,} exemplos")
print(f"   Teste (2025+):      {X_test.shape[0]:,} exemplos")

# 6. CRIAR DIVISÃO POR PORTE DE CIDADE
print("\n📊 Criando subsets por porte de cidade...")

try:
    with open('cidades_por_porte.json', 'r', encoding='utf-8') as f:
        cidades_por_porte = json.load(f)
    
    for porte in ['grande', 'media', 'pequena']:
        lista_cidades = cidades_por_porte.get(porte, [])
        
        # Treino
        mask_treino = np.isin(cidades_train, lista_cidades)
        X_porte_train = X_train[mask_treino]
        y_porte_train = y_train[mask_treino]
        cidades_id_porte_train = cidades_id_train[mask_treino]
        
        # Teste
        mask_teste = np.isin(cidades_test, lista_cidades)
        X_porte_test = X_test[mask_teste]
        y_porte_test = y_test[mask_teste]
        cidades_id_porte_test = cidades_id_test[mask_teste]
        cidades_porte_test = cidades_test[mask_teste]
        datas_porte_test = datas_test[mask_teste]
        
        if len(X_porte_train) > 0:
            np.save(f'X_train_{porte}.npy', X_porte_train)
            np.save(f'y_train_{porte}.npy', y_porte_train)
            np.save(f'cidades_id_train_{porte}.npy', cidades_id_porte_train)
            np.save(f'X_test_{porte}.npy', X_porte_test)
            np.save(f'y_test_{porte}.npy', y_porte_test)
            np.save(f'cidades_id_test_{porte}.npy', cidades_id_porte_test)
            np.save(f'cidades_test_{porte}.npy', cidades_porte_test)
            np.save(f'datas_test_{porte}.npy', datas_porte_test)
            
            print(f"   {porte.upper()}: Treino={len(X_porte_train):,}, Teste={len(X_porte_test):,}")

except FileNotFoundError:
    print("   ⚠️ Arquivo 'cidades_por_porte.json' não encontrado. Pulando divisão por porte.")

# 7. SALVAR DADOS COMPLETOS
print("\n💾 Salvando arquivos processados...")

np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)
np.save('cidades_id_train.npy', cidades_id_train)
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
np.save('cidades_id_test.npy', cidades_id_test)
np.save('cidades_test.npy', cidades_test)
np.save('datas_test.npy', datas_test)

# Salvar metadados
metadados = {
    'tamanho_janela': TAMANHO_JANELA,
    'colunas_features': COLUNAS_FEATURES,
    'coluna_alvo': COLUNA_ALVO,
    'data_corte': str(DATA_CORTE),
    'n_cidades': len(df['cidade'].unique()),
    'n_features': len(COLUNAS_FEATURES),
    'n_treino': int(X_train.shape[0]),
    'n_teste': int(X_test.shape[0])
}

with open('metadados.json', 'w', encoding='utf-8') as f:
    json.dump(metadados, f, ensure_ascii=False, indent=2)

# 8. CRIAR FOLDS PARA VALIDAÇÃO CRUZADA TEMPORAL
print("\n📅 Criando folds para validação cruzada temporal...")

# Divide os dados de treino em 5 períodos temporais
datas_treino_sorted = np.sort(np.unique(datas_train))
n_datas = len(datas_treino_sorted)
fold_size = n_datas // 5

folds = []
for i in range(5):
    start_idx = i * fold_size
    if i == 4:  # Último fold pega o resto
        end_idx = n_datas
    else:
        end_idx = (i + 1) * fold_size
    
    val_start = datas_treino_sorted[start_idx]
    val_end = datas_treino_sorted[min(end_idx - 1, n_datas - 1)]
    
    folds.append({
        'fold': i + 1,
        'val_start': str(val_start),
        'val_end': str(val_end)
    })
    print(f"   Fold {i+1}: {val_start} a {val_end}")

with open('folds_temporais.json', 'w', encoding='utf-8') as f:
    json.dump(folds, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("✅ CRIAÇÃO DE SEQUÊNCIAS DENGO CONCLUÍDA!")
print("=" * 60)
print(f"\n📋 RESUMO:")
print(f"   • Janela temporal: {TAMANHO_JANELA} semanas")
print(f"   • Features: {len(COLUNAS_FEATURES)}")
print(f"   • Normalização: RobustScaler por cidade")
print(f"   • Exemplos de treino: {X_train.shape[0]:,}")
print(f"   • Exemplos de teste: {X_test.shape[0]:,}")
print(f"\n📁 Arquivos salvos:")
print(f"   • X_train.npy, y_train.npy (dados de treino)")
print(f"   • X_test.npy, y_test.npy (dados de teste)")
print(f"   • scalers_por_cidade.pkl (para desnormalização)")
print(f"   • label_encoder_cidades.pkl (encoding de cidades)")
print(f"   • metadados.json (configurações)")