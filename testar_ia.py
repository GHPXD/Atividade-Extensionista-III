"""
TESTAR IA DENGO - Avaliação Completa
=================================
Funcionalidades:
1. ✅ Métricas completas: R², RMSE, MAE, MAPE
2. ✅ Avaliação por cidade
3. ✅ Comparação com múltiplos baselines
4. ✅ Análise de erros por faixa de casos
5. ✅ Gráficos detalhados
6. ✅ Desnormalização correta por cidade
7. ✅ Relatório exportável em JSON/CSV
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURAÇÕES ---
ARQUIVO_MODELO = "dengo_ai.keras"
ARQUIVO_DATASET = "DATASET_PARA_IA.csv"
CIDADES_ANALISE = None  # None = todas, ou lista: ["Curitiba", "Londrina"]

print("=" * 70)
print("🔬 AVALIAÇÃO COMPLETA DO MODELO DENGO")
print("=" * 70)

# 1. CARREGAR COMPONENTES
print("\n📂 Carregando componentes...")

try:
    model = tf.keras.models.load_model(ARQUIVO_MODELO)
    print(f"   ✅ Modelo carregado: {ARQUIVO_MODELO}")
except Exception as e:
    print(f"   ❌ Erro ao carregar modelo: {e}")
    exit()

try:
    scalers_por_cidade = joblib.load('scalers_por_cidade.pkl')
    print(f"   ✅ Scalers carregados: {len(scalers_por_cidade)} cidades")
except:
    print("   ⚠️ Scalers por cidade não encontrados")
    scalers_por_cidade = None

try:
    with open('metadados.json', 'r') as f:
        metadados = json.load(f)
    print(f"   ✅ Metadados carregados")
except:
    metadados = {'colunas_features': [], 'tamanho_janela': 12}

try:
    X_test = np.load('X_test.npy')
    y_test = np.load('y_test.npy')
    cidades_id_test = np.load('cidades_id_test.npy')
    cidades_test = np.load('cidades_test.npy', allow_pickle=True)
    datas_test = np.load('datas_test.npy', allow_pickle=True)
    print(f"   ✅ Dados de teste carregados: {len(X_test):,} amostras")
except Exception as e:
    print(f"   ❌ Erro ao carregar dados de teste: {e}")
    exit()

# 2. FAZER PREVISÕES
print("\n🔮 Gerando previsões...")
y_pred = model.predict([X_test, cidades_id_test], verbose=0).flatten()
y_pred = np.maximum(y_pred, 0)  # Garantir não-negativo

# 3. DESNORMALIZAR (se necessário)
# Nota: Os dados já estão normalizados por cidade, precisamos desnormalizar para métricas reais
print("\n🔄 Desnormalizando previsões...")

def desnormalizar_por_cidade(valores, cidades, scalers, col_idx=0):
    """Desnormaliza valores usando o scaler de cada cidade"""
    valores_desnorm = np.zeros_like(valores)
    
    for i, (valor, cidade) in enumerate(zip(valores, cidades)):
        if cidade in scalers:
            scaler = scalers[cidade]
            # Criar array dummy para inverse_transform
            dummy = np.zeros((1, scaler.n_features_in_))
            dummy[0, col_idx] = valor
            valores_desnorm[i] = scaler.inverse_transform(dummy)[0, col_idx]
        else:
            valores_desnorm[i] = valor  # Mantém normalizado se não tiver scaler
    
    return valores_desnorm

if scalers_por_cidade:
    y_test_real = desnormalizar_por_cidade(y_test, cidades_test, scalers_por_cidade)
    y_pred_real = desnormalizar_por_cidade(y_pred, cidades_test, scalers_por_cidade)
    print("   ✅ Valores desnormalizados para métricas reais")
else:
    y_test_real = y_test
    y_pred_real = y_pred
    print("   ⚠️ Usando valores normalizados (scaler não disponível)")

# 4. MÉTRICAS GLOBAIS
print("\n" + "=" * 70)
print("📊 MÉTRICAS GLOBAIS (Todas as cidades)")
print("=" * 70)

def calcular_metricas(y_real, y_previsto):
    """Calcula todas as métricas de avaliação"""
    # Evitar divisão por zero
    mask = y_real > 0
    
    r2 = r2_score(y_real, y_previsto)
    rmse = np.sqrt(mean_squared_error(y_real, y_previsto))
    mae = mean_absolute_error(y_real, y_previsto)
    
    # MAPE apenas onde y_real > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_real[mask] - y_previsto[mask]) / y_real[mask])) * 100
    else:
        mape = np.nan
    
    return {
        'R²': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE (%)': mape
    }

metricas_globais = calcular_metricas(y_test_real, y_pred_real)

print(f"\n   {'Métrica':<15} {'Valor':>15}")
print("   " + "-" * 32)
for metrica, valor in metricas_globais.items():
    print(f"   {metrica:<15} {valor:>15.4f}")

# 5. COMPARAÇÃO COM BASELINES
print("\n" + "=" * 70)
print("📈 COMPARAÇÃO COM BASELINES")
print("=" * 70)

# Baseline 1: Prever a média global
baseline_media = np.full_like(y_test_real, np.mean(y_test_real))
metricas_media = calcular_metricas(y_test_real, baseline_media)

# Baseline 2: Prever a última semana (semana anterior)
baseline_ultima_semana = X_test[:, -1, 0]  # Última semana, casos_est
if scalers_por_cidade:
    baseline_ultima_semana = desnormalizar_por_cidade(
        baseline_ultima_semana, cidades_test, scalers_por_cidade
    )
metricas_ultima = calcular_metricas(y_test_real, baseline_ultima_semana)

# Baseline 3: Média móvel das últimas 4 semanas
baseline_ma4 = X_test[:, -4:, 0].mean(axis=1)
if scalers_por_cidade:
    baseline_ma4 = desnormalizar_por_cidade(baseline_ma4, cidades_test, scalers_por_cidade)
metricas_ma4 = calcular_metricas(y_test_real, baseline_ma4)

print(f"\n   {'Modelo/Baseline':<25} {'R²':>10} {'RMSE':>12} {'MAE':>10}")
print("   " + "-" * 60)
print(f"   {'🤖 Modelo Dengo':<25} {metricas_globais['R²']:>10.4f} {metricas_globais['RMSE']:>12.2f} {metricas_globais['MAE']:>10.2f}")
print(f"   {'📉 Média Global':<25} {metricas_media['R²']:>10.4f} {metricas_media['RMSE']:>12.2f} {metricas_media['MAE']:>10.2f}")
print(f"   {'📅 Semana Anterior':<25} {metricas_ultima['R²']:>10.4f} {metricas_ultima['RMSE']:>12.2f} {metricas_ultima['MAE']:>10.2f}")
print(f"   {'📊 Média Móvel 4 sem.':<25} {metricas_ma4['R²']:>10.4f} {metricas_ma4['RMSE']:>12.2f} {metricas_ma4['MAE']:>10.2f}")

# 6. MÉTRICAS POR CIDADE
print("\n" + "=" * 70)
print("🏙️ MÉTRICAS POR CIDADE (Top 20 maiores erros e menores erros)")
print("=" * 70)

metricas_por_cidade = {}

for cidade in np.unique(cidades_test):
    mask = cidades_test == cidade
    if mask.sum() < 5:  # Poucas amostras
        continue
    
    y_real_cidade = y_test_real[mask]
    y_pred_cidade = y_pred_real[mask]
    
    metricas = calcular_metricas(y_real_cidade, y_pred_cidade)
    metricas['n_amostras'] = int(mask.sum())
    metricas['media_casos'] = float(np.mean(y_real_cidade))
    metricas_por_cidade[cidade] = metricas

# Ordenar por R²
cidades_ordenadas = sorted(metricas_por_cidade.keys(), key=lambda x: metricas_por_cidade[x]['R²'], reverse=True)

print("\n   🏆 TOP 10 MELHORES PREVISÕES:")
print(f"   {'Cidade':<25} {'R²':>8} {'MAE':>10} {'Casos Médios':>12}")
print("   " + "-" * 58)
for cidade in cidades_ordenadas[:10]:
    m = metricas_por_cidade[cidade]
    print(f"   {cidade:<25} {m['R²']:>8.3f} {m['MAE']:>10.1f} {m['media_casos']:>12.1f}")

print("\n   ⚠️ TOP 10 PIORES PREVISÕES:")
print(f"   {'Cidade':<25} {'R²':>8} {'MAE':>10} {'Casos Médios':>12}")
print("   " + "-" * 58)
for cidade in cidades_ordenadas[-10:]:
    m = metricas_por_cidade[cidade]
    print(f"   {cidade:<25} {m['R²']:>8.3f} {m['MAE']:>10.1f} {m['media_casos']:>12.1f}")

# 7. ANÁLISE POR FAIXA DE CASOS
print("\n" + "=" * 70)
print("📊 ANÁLISE POR FAIXA DE CASOS")
print("=" * 70)

faixas = [
    (0, 5, 'Muito baixo (0-5)'),
    (5, 20, 'Baixo (5-20)'),
    (20, 50, 'Médio (20-50)'),
    (50, 100, 'Alto (50-100)'),
    (100, float('inf'), 'Muito alto (100+)')
]

print(f"\n   {'Faixa':<20} {'Amostras':>10} {'R²':>8} {'MAE':>10} {'MAPE %':>10}")
print("   " + "-" * 62)

for min_val, max_val, nome in faixas:
    mask = (y_test_real >= min_val) & (y_test_real < max_val)
    if mask.sum() > 10:
        metricas_faixa = calcular_metricas(y_test_real[mask], y_pred_real[mask])
        print(f"   {nome:<20} {mask.sum():>10,} {metricas_faixa['R²']:>8.3f} {metricas_faixa['MAE']:>10.1f} {metricas_faixa['MAPE (%)']:>10.1f}")

# 8. GRÁFICOS
print("\n📈 Gerando gráficos de avaliação...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Gráfico 1: Scatter Real vs Previsto
ax = axes[0, 0]
sample_idx = np.random.choice(len(y_test_real), min(10000, len(y_test_real)), replace=False)
ax.scatter(y_test_real[sample_idx], y_pred_real[sample_idx], alpha=0.3, s=1)
max_val = max(y_test_real.max(), y_pred_real.max())
ax.plot([0, max_val], [0, max_val], 'r--', label='Ideal')
ax.set_xlabel('Casos Reais')
ax.set_ylabel('Casos Previstos')
ax.set_title(f'Real vs Previsto (R² = {metricas_globais["R²"]:.4f})')
ax.legend()
ax.grid(True, alpha=0.3)

# Gráfico 2: Distribuição dos Erros
ax = axes[0, 1]
erros = y_test_real - y_pred_real
ax.hist(erros, bins=100, edgecolor='black', alpha=0.7)
ax.axvline(x=0, color='r', linestyle='--')
ax.set_xlabel('Erro (Real - Previsto)')
ax.set_ylabel('Frequência')
ax.set_title(f'Distribuição dos Erros (μ={np.mean(erros):.2f}, σ={np.std(erros):.2f})')
ax.grid(True, alpha=0.3)

# Gráfico 3: R² por Cidade (histograma)
ax = axes[0, 2]
r2_cidades = [metricas_por_cidade[c]['R²'] for c in metricas_por_cidade]
ax.hist(r2_cidades, bins=30, edgecolor='black', alpha=0.7)
ax.axvline(x=np.median(r2_cidades), color='r', linestyle='--', label=f'Mediana: {np.median(r2_cidades):.3f}')
ax.axvline(x=0, color='orange', linestyle='--', label='R²=0 (baseline)')
ax.set_xlabel('R² por Cidade')
ax.set_ylabel('Número de Cidades')
ax.set_title('Distribuição de R² por Cidade')
ax.legend()
ax.grid(True, alpha=0.3)

# Gráfico 4: Séries temporais para Curitiba (se disponível)
ax = axes[1, 0]
cidade_exemplo = 'Curitiba' if 'Curitiba' in np.unique(cidades_test) else np.unique(cidades_test)[0]
mask_cidade = cidades_test == cidade_exemplo
if mask_cidade.sum() > 0:
    datas_cidade = datas_test[mask_cidade]
    real_cidade = y_test_real[mask_cidade]
    pred_cidade = y_pred_real[mask_cidade]
    
    # Ordenar por data
    sort_idx = np.argsort(datas_cidade)
    ax.plot(range(len(real_cidade)), real_cidade[sort_idx], label='Real', alpha=0.7)
    ax.plot(range(len(pred_cidade)), pred_cidade[sort_idx], label='Previsto', alpha=0.7, linestyle='--')
    ax.set_xlabel('Semanas (2025)')
    ax.set_ylabel('Casos')
    ax.set_title(f'Série Temporal - {cidade_exemplo}')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Gráfico 5: Erro percentual por faixa
ax = axes[1, 1]
faixas_labels = []
mapes = []
for min_val, max_val, nome in faixas:
    mask = (y_test_real >= min_val) & (y_test_real < max_val)
    if mask.sum() > 10:
        mask_nonzero = mask & (y_test_real > 0)
        if mask_nonzero.sum() > 0:
            mape = np.mean(np.abs((y_test_real[mask_nonzero] - y_pred_real[mask_nonzero]) / y_test_real[mask_nonzero])) * 100
            faixas_labels.append(nome.split('(')[0].strip())
            mapes.append(mape)

ax.bar(faixas_labels, mapes, color='steelblue', edgecolor='black')
ax.set_xlabel('Faixa de Casos')
ax.set_ylabel('MAPE (%)')
ax.set_title('Erro Percentual por Faixa de Casos')
ax.tick_params(axis='x', rotation=15)
ax.grid(True, alpha=0.3, axis='y')

# Gráfico 6: Comparação de R² (Modelo vs Baselines)
ax = axes[1, 2]
modelos = ['Modelo Dengo', 'Média Global', 'Sem. Anterior', 'Média Móvel 4']
r2_valores = [
    metricas_globais['R²'], 
    metricas_media['R²'], 
    metricas_ultima['R²'], 
    metricas_ma4['R²']
]
colors = ['green' if r > 0 else 'red' for r in r2_valores]
ax.bar(modelos, r2_valores, color=colors, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.set_ylabel('R²')
ax.set_title('Comparação com Baselines')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('avaliacao_modelo.png', dpi=150)
print("   ✅ Gráfico salvo: avaliacao_modelo.png")

# 9. EXPORTAR RELATÓRIO
print("\n📄 Exportando relatório...")

relatorio = {
    'data_avaliacao': str(np.datetime64('today')),
    'modelo': ARQUIVO_MODELO,
    'metricas_globais': metricas_globais,
    'comparacao_baselines': {
        'modelo_dengo': metricas_globais,
        'media_global': metricas_media,
        'semana_anterior': metricas_ultima,
        'media_movel_4': metricas_ma4
    },
    'metricas_por_cidade': metricas_por_cidade,
    'resumo': {
        'total_amostras': int(len(y_test)),
        'total_cidades': len(metricas_por_cidade),
        'r2_mediano_cidades': float(np.median(r2_cidades)),
        'cidades_r2_positivo': int(sum(1 for r in r2_cidades if r > 0)),
        'cidades_r2_negativo': int(sum(1 for r in r2_cidades if r <= 0))
    }
}

with open('relatorio_avaliacao.json', 'w', encoding='utf-8') as f:
    json.dump(relatorio, f, ensure_ascii=False, indent=2, default=str)
print("   ✅ Relatório salvo: relatorio_avaliacao.json")

# Salvar métricas por cidade em CSV
df_metricas = pd.DataFrame(metricas_por_cidade).T
df_metricas.index.name = 'cidade'
df_metricas.to_csv('metricas_por_cidade.csv')
print("   ✅ Métricas por cidade: metricas_por_cidade.csv")

# 10. RESUMO FINAL
print("\n" + "=" * 70)
print("📋 RESUMO DA AVALIAÇÃO")
print("=" * 70)

print(f"""
   📊 MÉTRICAS GLOBAIS:
      • R²:   {metricas_globais['R²']:.4f}
      • RMSE: {metricas_globais['RMSE']:.2f} casos
      • MAE:  {metricas_globais['MAE']:.2f} casos
      • MAPE: {metricas_globais['MAPE (%)']:.1f}%

   📈 COMPARAÇÃO COM BASELINES:
      • Modelo Dengo {'✅ MELHOR' if metricas_globais['R²'] > metricas_ultima['R²'] else '❌ PIOR'} que "semana anterior"
      • Melhoria sobre baseline: {(metricas_globais['R²'] - metricas_ultima['R²']) * 100:.1f}% em R²

   🏙️ ANÁLISE POR CIDADE:
      • Cidades com R² > 0: {relatorio['resumo']['cidades_r2_positivo']} ({100*relatorio['resumo']['cidades_r2_positivo']/len(metricas_por_cidade):.1f}%)
      • Cidades com R² ≤ 0: {relatorio['resumo']['cidades_r2_negativo']} ({100*relatorio['resumo']['cidades_r2_negativo']/len(metricas_por_cidade):.1f}%)
      • R² mediano: {relatorio['resumo']['r2_mediano_cidades']:.4f}

   💡 RECOMENDAÇÕES:
""")

if metricas_globais['R²'] > 0.5:
    print("      ✅ Modelo está bom para produção!")
elif metricas_globais['R²'] > 0:
    print("      ⚠️ Modelo precisa de ajustes, mas é melhor que baseline")
else:
    print("      ❌ Modelo precisa de revisão significativa")
    print("      • Considere treinar modelos separados por porte de cidade")
    print("      • Verifique se há mais data leakage")
    print("      • Considere aumentar a complexidade do modelo")

print("\n" + "=" * 70)
print("✅ AVALIAÇÃO CONCLUÍDA!")
print("=" * 70)