"""
PREPARAR DATASET DENGO - Versão Principal
======================================
Implementações:
1. ✅ Preenchimento com média POR CIDADE (não global)
2. ✅ Detecção e tratamento de outliers
3. ✅ REMOÇÃO da coluna Rt (data leakage!)
4. ✅ Classificação de cidades por porte (para modelos separados)
5. ✅ Validação de qualidade dos dados
"""

import pandas as pd
import glob
import os
import numpy as np
from collections import defaultdict

# --- CONFIGURAÇÕES ---
PASTA_RAIZ = "dados_dengue"
ARQUIVO_SAIDA = "DATASET_PARA_IA.csv"

# Limiares para classificação de cidades
POPULACAO_GRANDE = 200_000  # > 200K = cidade grande
POPULACAO_MEDIA = 50_000    # 50K-200K = cidade média
# < 50K = cidade pequena

print("=" * 60)
print("🚀 PREPARAÇÃO DE DADOS DENGO - VERSÃO PRINCIPAL")
print("=" * 60)

# 1. LISTAR TODOS OS ARQUIVOS
arquivos = glob.glob(os.path.join(PASTA_RAIZ, "**", "*.csv"), recursive=True)
print(f"\n📊 Total de arquivos encontrados: {len(arquivos)}")

lista_dfs = []
estatisticas = defaultdict(int)

# 2. LER E JUNTAR
for i, arquivo in enumerate(arquivos):
    try:
        df_temp = pd.read_csv(arquivo)
        nome_cidade = os.path.basename(arquivo).replace(".csv", "")
        df_temp['cidade'] = nome_cidade
        
        # Selecionar apenas as colunas úteis
        # ⚠️ NOTA: Removemos 'Rt' aqui pois é DATA LEAKAGE!
        colunas_uteis = [
            'data_iniSE', 'SE', 'casos_est', 'casos',  # Alvos
            'tempmed', 'tempmin', 'tempmax',           # Clima Temp
            'umidmed', 'umidmin', 'umidmax',           # Clima Umidade
            'receptivo', 'pop',                        # Epidemiologia (SEM Rt!)
            'cidade'
        ]
        
        cols_existentes = [c for c in colunas_uteis if c in df_temp.columns]
        df_temp = df_temp[cols_existentes]
        lista_dfs.append(df_temp)
        
        estatisticas['arquivos_lidos'] += 1
        
        if i % 500 == 0 and i > 0:
            print(f"   Processado: {i}/{len(arquivos)}")
            
    except Exception as e:
        estatisticas['arquivos_erro'] += 1
        print(f"⚠️ Erro no arquivo {arquivo}: {e}")

if not lista_dfs:
    print("❌ Nenhum arquivo foi carregado! Verifique a pasta 'dados_dengue'.")
    exit()

print("🔄 Mesclando dataframes...")
df_final = pd.concat(lista_dfs, ignore_index=True)
print(f"   Total de linhas brutas: {len(df_final)}")

# 3. TRATAMENTO DE DADOS

# A. Converter data (Unix Timestamp -> Datetime)
print("\n🧹 Realizando limpeza e tratamento...")
if 'data_iniSE' in df_final.columns:
    df_final['data'] = pd.to_datetime(df_final['data_iniSE'], unit='ms')

# B. Ordenação Rigorosa (Cidade -> Data)
df_final.sort_values(by=['cidade', 'data'], inplace=True)

# C. Preencher casos com 0 se NaN
df_final['casos_est'] = df_final['casos_est'].fillna(0)
df_final['casos'] = df_final['casos'].fillna(0)

# D. CORREÇÃO CLIMÁTICA POR CIDADE (não global!)
cols_clima = ['tempmed', 'tempmin', 'tempmax', 'umidmed', 'umidmin', 'umidmax', 'receptivo']
cols_presentes = [c for c in cols_clima if c in df_final.columns]

print("   🔧 Preenchendo lacunas climáticas (Forward Fill por cidade)...")
df_final[cols_presentes] = df_final.groupby('cidade')[cols_presentes].ffill().bfill()

# E. FALLBACK: Se ainda sobrar NaN, usar média DA PRÓPRIA CIDADE
print("   🔧 Aplicando fallback com média por cidade (não global)...")
for col in cols_presentes:
    # Calcula a média por cidade
    medias_por_cidade = df_final.groupby('cidade')[col].transform('mean')
    # Preenche NaNs com a média da própria cidade
    mask_nan = df_final[col].isnull()
    df_final.loc[mask_nan, col] = medias_por_cidade[mask_nan]
    
    # Se ainda tiver NaN (cidade sem nenhum dado), usa média global como último recurso
    nans_restantes = df_final[col].isnull().sum()
    if nans_restantes > 0:
        media_global = df_final[col].mean()
        df_final[col] = df_final[col].fillna(media_global)
        print(f"      ⚠️ {col}: {nans_restantes} NaNs preenchidos com média global ({media_global:.2f})")

# 4. DETECÇÃO E TRATAMENTO DE OUTLIERS
print("\n🔍 Detectando outliers...")

def tratar_outliers(df, coluna, n_std=3):
    """Remove outliers usando método do z-score por cidade"""
    outliers_total = 0
    
    for cidade in df['cidade'].unique():
        mask_cidade = df['cidade'] == cidade
        indices_cidade = df.index[mask_cidade]
        dados = df.loc[indices_cidade, coluna]
        
        if len(dados) < 10:  # Poucas amostras, não dá pra calcular
            continue
            
        media = dados.mean()
        std = dados.std()
        
        if std > 0:
            # Identifica outliers (valores > 3 desvios padrão)
            limite_inferior = media - n_std * std
            limite_superior = media + n_std * std
            
            outliers = (dados < limite_inferior) | (dados > limite_superior)
            n_outliers = outliers.sum()
            
            if n_outliers > 0:
                # Substitui outliers pela mediana da cidade
                mediana = dados.median()
                # Usa os índices dos outliers diretamente
                indices_outliers = dados[outliers].index
                df.loc[indices_outliers, coluna] = mediana
                outliers_total += n_outliers
    
    return outliers_total

# Aplica detecção de outliers em colunas climáticas
for col in ['tempmed', 'tempmin', 'tempmax']:
    if col in df_final.columns:
        n_outliers = tratar_outliers(df_final, col, n_std=3)
        if n_outliers > 0:
            print(f"   ⚠️ {col}: {n_outliers} outliers corrigidos")

# Validação de temperatura (limite físico razoável para PR)
if 'tempmax' in df_final.columns:
    mascara_temp_invalida = df_final['tempmax'] > 50  # > 50°C é erro
    n_invalidos = mascara_temp_invalida.sum()
    if n_invalidos > 0:
        df_final.loc[mascara_temp_invalida, 'tempmax'] = df_final.loc[~mascara_temp_invalida, 'tempmax'].mean()
        print(f"   ⚠️ tempmax: {n_invalidos} valores > 50°C corrigidos")

# 5. CLASSIFICAÇÃO DE CIDADES POR PORTE
print("\n📊 Classificando cidades por porte...")

def classificar_cidade(pop):
    if pd.isna(pop):
        return 'pequena'
    elif pop >= POPULACAO_GRANDE:
        return 'grande'
    elif pop >= POPULACAO_MEDIA:
        return 'media'
    else:
        return 'pequena'

# Pega a população mais recente de cada cidade
pop_por_cidade = df_final.groupby('cidade')['pop'].last()
classificacao = pop_por_cidade.apply(classificar_cidade)

# Adiciona a classificação ao dataframe
df_final['porte_cidade'] = df_final['cidade'].map(classificacao)

# Conta cidades por porte
contagem_porte = df_final.groupby('cidade')['porte_cidade'].first().value_counts()
print(f"   Cidades grandes (>200K hab): {contagem_porte.get('grande', 0)}")
print(f"   Cidades médias (50K-200K):   {contagem_porte.get('media', 0)}")
print(f"   Cidades pequenas (<50K):     {contagem_porte.get('pequena', 0)}")

# Lista de cidades grandes (top 20 para modelo separado)
cidades_grandes = pop_por_cidade[pop_por_cidade >= POPULACAO_GRANDE].sort_values(ascending=False)
print(f"\n   🏙️ Top 10 cidades por população:")
for i, (cidade, pop) in enumerate(cidades_grandes.head(10).items()):
    print(f"      {i+1}. {cidade}: {int(pop):,} hab")

# 6. ESTATÍSTICAS FINAIS
print("\n📈 Estatísticas do Dataset Final:")
print(f"   Total de linhas: {len(df_final):,}")
print(f"   Total de cidades: {df_final['cidade'].nunique()}")
print(f"   Período: {df_final['data'].min().date()} a {df_final['data'].max().date()}")
print(f"   Colunas: {list(df_final.columns)}")

# Verificação de NaNs
nans_por_coluna = df_final.isnull().sum()
colunas_com_nan = nans_por_coluna[nans_por_coluna > 0]
if len(colunas_com_nan) > 0:
    print(f"\n   ⚠️ Colunas ainda com NaN:")
    for col, n in colunas_com_nan.items():
        print(f"      {col}: {n}")
else:
    print("   ✅ Nenhuma coluna com valores NaN!")

# 7. SALVAR
print(f"\n💾 Salvando arquivo final: {ARQUIVO_SAIDA}...")
df_final.to_csv(ARQUIVO_SAIDA, index=False)

# Salvar também lista de cidades por porte (útil para treinar modelos separados)
cidades_por_porte = {
    'grande': list(cidades_grandes.index),
    'media': list(pop_por_cidade[(pop_por_cidade >= POPULACAO_MEDIA) & (pop_por_cidade < POPULACAO_GRANDE)].index),
    'pequena': list(pop_por_cidade[pop_por_cidade < POPULACAO_MEDIA].index)
}

import json
with open('cidades_por_porte.json', 'w', encoding='utf-8') as f:
    json.dump(cidades_por_porte, f, ensure_ascii=False, indent=2)
print("   📁 Lista de cidades por porte salva em 'cidades_por_porte.json'")

print("\n" + "=" * 60)
print("✅ PREPARAÇÃO DENGO CONCLUÍDA!")
print("=" * 60)
print(f"\n📋 RESUMO:")
print(f"   • Arquivos processados: {estatisticas['arquivos_lidos']}")
print(f"   • Arquivos com erro: {estatisticas['arquivos_erro']}")
print(f"   • Coluna Rt REMOVIDA (data leakage)")
print(f"   • Outliers tratados por cidade")
print(f"   • Preenchimento de NaN por cidade (não global)")