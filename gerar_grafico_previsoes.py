"""
GRÁFICO DE PREVISÕES VS REALIDADE - 2025
========================================
Mostra a série temporal de casos reais vs previsões da IA
para as principais cidades do Paraná em 2025.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURAÇÕES ---
CIDADES_DESTAQUE = [
    'Curitiba', 'Londrina', 'Maringá', 'Cascavel', 
    'Foz do Iguaçu', 'Ponta Grossa', 'Toledo', 'Apucarana'
]

print("=" * 70)
print("📊 GERANDO GRÁFICOS: PREVISÃO DA IA vs REALIDADE (2025)")
print("=" * 70)

# 1. CARREGAR COMPONENTES
print("\n📂 Carregando componentes...")

model = tf.keras.models.load_model('dengo_ai.keras')
scalers_por_cidade = joblib.load('scalers_por_cidade.pkl')

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')
cidades_id_test = np.load('cidades_id_test.npy')
cidades_test = np.load('cidades_test.npy', allow_pickle=True)
datas_test = np.load('datas_test.npy', allow_pickle=True)

with open('metadados.json', 'r') as f:
    metadados = json.load(f)

print(f"   ✅ {len(X_test):,} amostras de teste carregadas")

# 2. FAZER PREVISÕES
print("\n🔮 Gerando previsões...")
y_pred = model.predict([X_test, cidades_id_test], verbose=0).flatten()
y_pred = np.maximum(y_pred, 0)

# 3. DESNORMALIZAR
print("🔄 Desnormalizando valores...")

def desnormalizar(valores, cidades, scalers, col_idx=0):
    """Desnormaliza valores usando o scaler de cada cidade"""
    valores_desnorm = np.zeros_like(valores)
    
    for i, (valor, cidade) in enumerate(zip(valores, cidades)):
        if cidade in scalers:
            scaler = scalers[cidade]
            dummy = np.zeros((1, scaler.n_features_in_))
            dummy[0, col_idx] = valor
            valores_desnorm[i] = scaler.inverse_transform(dummy)[0, col_idx]
        else:
            valores_desnorm[i] = valor
    
    return valores_desnorm

y_test_real = desnormalizar(y_test, cidades_test, scalers_por_cidade)
y_pred_real = desnormalizar(y_pred, cidades_test, scalers_por_cidade)

# 4. FILTRAR CIDADES DISPONÍVEIS
cidades_disponiveis = [c for c in CIDADES_DESTAQUE if c in np.unique(cidades_test)]
print(f"   Cidades para análise: {cidades_disponiveis}")

# 5. GRÁFICO PRINCIPAL - MÚLTIPLAS CIDADES
print("\n📈 Gerando gráfico principal...")

n_cidades = len(cidades_disponiveis)
fig, axes = plt.subplots(4, 2, figsize=(16, 20))
axes = axes.flatten()

for idx, cidade in enumerate(cidades_disponiveis):
    ax = axes[idx]
    
    # Filtrar dados da cidade
    mask = cidades_test == cidade
    datas_cidade = pd.to_datetime(datas_test[mask])
    real_cidade = y_test_real[mask]
    pred_cidade = y_pred_real[mask]
    
    # Ordenar por data
    df_cidade = pd.DataFrame({
        'data': datas_cidade,
        'real': real_cidade,
        'pred': pred_cidade
    }).sort_values('data')
    
    # Plotar
    ax.fill_between(df_cidade['data'], df_cidade['real'], alpha=0.3, color='blue', label='Casos Reais')
    ax.plot(df_cidade['data'], df_cidade['real'], color='blue', linewidth=2, label='_nolegend_')
    ax.plot(df_cidade['data'], df_cidade['pred'], color='red', linewidth=2, linestyle='--', label='Previsão IA')
    
    # Calcular métricas
    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(df_cidade['real'], df_cidade['pred'])
    mae = mean_absolute_error(df_cidade['real'], df_cidade['pred'])
    
    # Configurações do gráfico
    ax.set_title(f'{cidade}\nR² = {r2:.3f} | MAE = {mae:.1f} casos', fontsize=12, fontweight='bold')
    ax.set_xlabel('Data (2025)')
    ax.set_ylabel('Casos de Dengue')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    # Destacar picos
    pico_real = df_cidade['real'].max()
    pico_pred = df_cidade['pred'].max()
    
    ax.axhline(y=pico_real, color='blue', linestyle=':', alpha=0.5)
    ax.axhline(y=pico_pred, color='red', linestyle=':', alpha=0.5)

plt.suptitle('🦟 DENGUE NO PARANÁ 2025\nPrevisão da IA vs Casos Reais\n(IA treinada apenas com dados 2015-2024)', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('grafico_previsoes_2025.png', dpi=150, bbox_inches='tight')
print("   ✅ Salvo: grafico_previsoes_2025.png")

# 6. GRÁFICO AGREGADO - TOTAL DO PARANÁ
print("\n📊 Gerando gráfico agregado do estado...")

# Agrupar por semana
df_total = pd.DataFrame({
    'data': pd.to_datetime(datas_test),
    'real': y_test_real,
    'pred': y_pred_real
})

df_semanal = df_total.groupby('data').agg({
    'real': 'sum',
    'pred': 'sum'
}).reset_index().sort_values('data')

fig, ax = plt.subplots(figsize=(14, 7))

# Plotar
ax.fill_between(df_semanal['data'], df_semanal['real'], alpha=0.3, color='blue')
ax.plot(df_semanal['data'], df_semanal['real'], color='blue', linewidth=2.5, label='Casos Reais')
ax.plot(df_semanal['data'], df_semanal['pred'], color='red', linewidth=2.5, linestyle='--', label='Previsão da IA')

# Métricas
r2_total = r2_score(df_semanal['real'], df_semanal['pred'])
mae_total = mean_absolute_error(df_semanal['real'], df_semanal['pred'])

ax.set_title(f'🦟 DENGUE NO PARANÁ 2025 - TOTAL DO ESTADO\n'
             f'Previsão da IA vs Casos Reais | R² = {r2_total:.3f} | MAE = {mae_total:.0f} casos/semana',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Data', fontsize=12)
ax.set_ylabel('Total de Casos no Estado', fontsize=12)
ax.legend(loc='upper right', fontsize=12)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))
ax.xaxis.set_major_locator(mdates.MonthLocator())

# Anotações
pico_idx = df_semanal['real'].idxmax()
data_pico = df_semanal.loc[pico_idx, 'data']
valor_pico = df_semanal.loc[pico_idx, 'real']
ax.annotate(f'Pico: {valor_pico:,.0f} casos', 
            xy=(data_pico, valor_pico), 
            xytext=(data_pico, valor_pico * 1.1),
            fontsize=10, ha='center',
            arrowprops=dict(arrowstyle='->', color='blue'))

plt.tight_layout()
plt.savefig('grafico_previsoes_2025_total.png', dpi=150, bbox_inches='tight')
print("   ✅ Salvo: grafico_previsoes_2025_total.png")

# 7. GRÁFICO DE CORRELAÇÃO
print("\n📊 Gerando gráfico de correlação...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Scatter plot geral
ax = axes[0]
sample = np.random.choice(len(y_test_real), min(5000, len(y_test_real)), replace=False)
ax.scatter(y_test_real[sample], y_pred_real[sample], alpha=0.3, s=10, c='steelblue')
max_val = max(y_test_real.max(), y_pred_real.max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Previsão Perfeita')
ax.set_xlabel('Casos Reais', fontsize=12)
ax.set_ylabel('Casos Previstos pela IA', fontsize=12)
ax.set_title(f'Correlação: Real vs Previsto\nR² = {r2_score(y_test_real, y_pred_real):.4f}', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Distribuição dos erros
ax = axes[1]
erros = y_test_real - y_pred_real
ax.hist(erros, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Erro Zero')
ax.axvline(x=np.mean(erros), color='orange', linestyle='-', linewidth=2, label=f'Média: {np.mean(erros):.1f}')
ax.set_xlabel('Erro (Real - Previsto)', fontsize=12)
ax.set_ylabel('Frequência', fontsize=12)
ax.set_title('Distribuição dos Erros de Previsão', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Série temporal do erro
ax = axes[2]
df_erro = pd.DataFrame({
    'data': pd.to_datetime(datas_test),
    'erro': np.abs(y_test_real - y_pred_real)
})
df_erro_semanal = df_erro.groupby('data')['erro'].mean().reset_index().sort_values('data')
ax.plot(df_erro_semanal['data'], df_erro_semanal['erro'], color='orange', linewidth=2)
ax.fill_between(df_erro_semanal['data'], df_erro_semanal['erro'], alpha=0.3, color='orange')
ax.set_xlabel('Data', fontsize=12)
ax.set_ylabel('Erro Absoluto Médio', fontsize=12)
ax.set_title('Evolução do Erro ao Longo de 2025', fontsize=12)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.grid(True, alpha=0.3)

plt.suptitle('📊 ANÁLISE DE QUALIDADE DAS PREVISÕES - DENGUE PR 2025', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('grafico_analise_previsoes.png', dpi=150, bbox_inches='tight')
print("   ✅ Salvo: grafico_analise_previsoes.png")

# 8. RESUMO
print("\n" + "=" * 70)
print("✅ GRÁFICOS GERADOS COM SUCESSO!")
print("=" * 70)
print(f"""
📁 Arquivos salvos:
   • grafico_previsoes_2025.png      - Comparação por cidade
   • grafico_previsoes_2025_total.png - Total do estado
   • grafico_analise_previsoes.png   - Análise de qualidade

📊 Métricas Gerais:
   • R² Global: {r2_score(y_test_real, y_pred_real):.4f}
   • MAE Global: {mean_absolute_error(y_test_real, y_pred_real):.2f} casos
   • Erro Médio: {np.mean(erros):.2f} casos (bias)

🎯 O modelo foi treinado APENAS com dados de 2015-2024.
   Os dados de 2025 são COMPLETAMENTE novos para a IA!
""")

plt.show()