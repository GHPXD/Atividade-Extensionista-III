"""
GRÁFICO DE PREVISÕES VS REALIDADE - 2025
========================================
Comparação entre cidades GRANDES e PEQUENAS
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURAÇÕES ---
CIDADES_GRANDES = [
    'Curitiba', 'Londrina', 'Maringá', 'Cascavel', 
    'Foz do Iguaçu', 'Ponta Grossa', 'Toledo', 'Apucarana'
]

CIDADES_PEQUENAS = [
    'Abatiá', 'Adrianópolis', 'Agudos do Sul', 'Altamira do Paraná',
    'Alto Paraíso', 'Anahy', 'Bela Vista da Caroba', 'Boa Esperança'
]

print("=" * 70)
print("📊 GERANDO GRÁFICOS: CIDADES GRANDES vs PEQUENAS (2025)")
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

print(f"   ✅ {len(X_test):,} amostras de teste carregadas")

# 2. FAZER PREVISÕES
print("\n🔮 Gerando previsões...")
y_pred = model.predict([X_test, cidades_id_test], verbose=0).flatten()
y_pred = np.maximum(y_pred, 0)

# 3. DESNORMALIZAR
print("🔄 Desnormalizando valores...")

def desnormalizar(valores, cidades, scalers, col_idx=0):
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

# 4. FUNÇÃO PARA PLOTAR GRID DE CIDADES
def plotar_grid_cidades(cidades_lista, titulo, nome_arquivo):
    """Plota grid de gráficos para uma lista de cidades"""
    
    # Filtrar cidades disponíveis
    cidades_disponiveis = [c for c in cidades_lista if c in np.unique(cidades_test)]
    
    if len(cidades_disponiveis) == 0:
        print(f"   ⚠️ Nenhuma cidade encontrada para {titulo}")
        return
    
    # Ajustar grid
    n = len(cidades_disponiveis)
    cols = 2
    rows = (n + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4*rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes = np.array(axes).flatten()
    
    metricas_cidades = []
    
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
        ax.fill_between(df_cidade['data'], df_cidade['real'], alpha=0.3, color='blue')
        ax.plot(df_cidade['data'], df_cidade['real'], color='blue', linewidth=2, label='Casos Reais')
        ax.plot(df_cidade['data'], df_cidade['pred'], color='red', linewidth=2, linestyle='--', label='Previsão IA')
        
        # Calcular métricas
        r2 = r2_score(df_cidade['real'], df_cidade['pred']) if df_cidade['real'].std() > 0 else 0
        mae = mean_absolute_error(df_cidade['real'], df_cidade['pred'])
        media_casos = df_cidade['real'].mean()
        
        metricas_cidades.append({
            'cidade': cidade,
            'r2': r2,
            'mae': mae,
            'media': media_casos
        })
        
        # Configurações
        cor_r2 = 'green' if r2 > 0.5 else 'orange' if r2 > 0 else 'red'
        ax.set_title(f'{cidade}\nR² = {r2:.3f} | MAE = {mae:.1f} | Média = {media_casos:.1f} casos', 
                     fontsize=11, fontweight='bold', color=cor_r2)
        ax.set_xlabel('Data (2025)')
        ax.set_ylabel('Casos de Dengue')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    
    # Esconder axes vazios
    for idx in range(len(cidades_disponiveis), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'🦟 {titulo}\nPrevisão da IA vs Casos Reais (2025)\nIA treinada apenas com dados 2015-2024', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(nome_arquivo, dpi=150, bbox_inches='tight')
    print(f"   ✅ Salvo: {nome_arquivo}")
    plt.close()
    
    return metricas_cidades

# 5. GERAR GRÁFICOS
print("\n📈 Gerando gráficos para CIDADES GRANDES...")
metricas_grandes = plotar_grid_cidades(
    CIDADES_GRANDES, 
    'CIDADES GRANDES (>200K hab)', 
    'grafico_cidades_grandes.png'
)

print("\n📈 Gerando gráficos para CIDADES PEQUENAS...")
metricas_pequenas = plotar_grid_cidades(
    CIDADES_PEQUENAS, 
    'CIDADES PEQUENAS (<50K hab)', 
    'grafico_cidades_pequenas.png'
)

# 6. GRÁFICO COMPARATIVO
print("\n📊 Gerando gráfico comparativo...")

# Calcular métricas por porte
with open('cidades_por_porte.json', 'r', encoding='utf-8') as f:
    cidades_por_porte = json.load(f)

def calcular_metricas_porte(lista_cidades):
    mask = np.isin(cidades_test, lista_cidades)
    if mask.sum() == 0:
        return {'r2': 0, 'mae': 0, 'media': 0, 'n': 0}
    
    y_real = y_test_real[mask]
    y_prev = y_pred_real[mask]
    
    return {
        'r2': r2_score(y_real, y_prev) if y_real.std() > 0 else 0,
        'mae': mean_absolute_error(y_real, y_prev),
        'media': y_real.mean(),
        'n': mask.sum()
    }

metricas_por_porte = {
    'Grandes': calcular_metricas_porte(cidades_por_porte.get('grande', [])),
    'Médias': calcular_metricas_porte(cidades_por_porte.get('media', [])),
    'Pequenas': calcular_metricas_porte(cidades_por_porte.get('pequena', []))
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Gráfico 1: R² por porte
ax = axes[0]
portes = list(metricas_por_porte.keys())
r2_valores = [metricas_por_porte[p]['r2'] for p in portes]
colors = ['green' if r > 0.5 else 'orange' if r > 0 else 'red' for r in r2_valores]
bars = ax.bar(portes, r2_valores, color=colors, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Bom (R²=0.5)')
ax.set_ylabel('R²')
ax.set_title('R² por Porte de Cidade', fontweight='bold')
ax.set_ylim(-0.5, 1)
for bar, val in zip(bars, r2_valores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
            f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Gráfico 2: MAE por porte
ax = axes[1]
mae_valores = [metricas_por_porte[p]['mae'] for p in portes]
bars = ax.bar(portes, mae_valores, color='steelblue', edgecolor='black')
ax.set_ylabel('MAE (casos)')
ax.set_title('Erro Absoluto Médio por Porte', fontweight='bold')
for bar, val in zip(bars, mae_valores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Gráfico 3: Média de casos por porte
ax = axes[2]
media_valores = [metricas_por_porte[p]['media'] for p in portes]
bars = ax.bar(portes, media_valores, color='coral', edgecolor='black')
ax.set_ylabel('Média de Casos/Semana')
ax.set_title('Média de Casos por Porte', fontweight='bold')
for bar, val in zip(bars, media_valores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('📊 COMPARAÇÃO POR PORTE DE CIDADE - DENGUE PR 2025', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('grafico_comparacao_portes.png', dpi=150, bbox_inches='tight')
print("   ✅ Salvo: grafico_comparacao_portes.png")
plt.close()

# 7. RESUMO
print("\n" + "=" * 70)
print("📋 RESUMO COMPARATIVO")
print("=" * 70)

print(f"""
┌─────────────────────────────────────────────────────────────────────┐
│                    MÉTRICAS POR PORTE DE CIDADE                     │
├───────────────┬──────────┬──────────┬────────────┬─────────────────┤
│ Porte         │ R²       │ MAE      │ Média Casos│ Amostras        │
├───────────────┼──────────┼──────────┼────────────┼─────────────────┤
│ 🏙️  Grandes   │ {metricas_por_porte['Grandes']['r2']:>8.4f} │ {metricas_por_porte['Grandes']['mae']:>8.2f} │ {metricas_por_porte['Grandes']['media']:>10.1f} │ {metricas_por_porte['Grandes']['n']:>15,} │
│ 🏘️  Médias    │ {metricas_por_porte['Médias']['r2']:>8.4f} │ {metricas_por_porte['Médias']['mae']:>8.2f} │ {metricas_por_porte['Médias']['media']:>10.1f} │ {metricas_por_porte['Médias']['n']:>15,} │
│ 🏠 Pequenas   │ {metricas_por_porte['Pequenas']['r2']:>8.4f} │ {metricas_por_porte['Pequenas']['mae']:>8.2f} │ {metricas_por_porte['Pequenas']['media']:>10.1f} │ {metricas_por_porte['Pequenas']['n']:>15,} │
└───────────────┴──────────┴──────────┴────────────┴─────────────────┘

💡 ANÁLISE:
""")

if metricas_por_porte['Grandes']['r2'] > 0.8:
    print("   ✅ Cidades GRANDES: Excelente performance (R² > 0.8)")
elif metricas_por_porte['Grandes']['r2'] > 0.5:
    print("   ✅ Cidades GRANDES: Boa performance (R² > 0.5)")
else:
    print("   ⚠️ Cidades GRANDES: Precisa melhorias")

if metricas_por_porte['Médias']['r2'] > 0.5:
    print("   ✅ Cidades MÉDIAS: Boa performance")
else:
    print("   ⚠️ Cidades MÉDIAS: Performance moderada")

if metricas_por_porte['Pequenas']['r2'] > 0:
    print("   ⚠️ Cidades PEQUENAS: R² positivo mas baixo (difícil prever 0-5 casos)")
else:
    print("   ❌ Cidades PEQUENAS: R² negativo - usar fallback recomendado")

print(f"""
📁 Gráficos salvos:
   • grafico_cidades_grandes.png
   • grafico_cidades_pequenas.png
   • grafico_comparacao_portes.png
""")