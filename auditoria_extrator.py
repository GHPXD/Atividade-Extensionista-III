import pandas as pd
import requests
import os
import time

# --- CONFIGURAÇÕES GOVTECH ---
ano_final = 2025
base_dir = "dados_dengue_brasil"

REGIOES = {
    "1": {"nome": "Norte", "id_ibge": 1},
    "2": {"nome": "Nordeste", "id_ibge": 2},
    "3": {"nome": "Sudeste", "id_ibge": 3},
    "4": {"nome": "Sul", "id_ibge": 4},
    "5": {"nome": "Centro-Oeste", "id_ibge": 5}
}

print("#" * 60)
print(" 🕵️ SISTEMA DE AUDITORIA E RESGATE - INFODENGUE")
print("#" * 60)

# --- 1. MENU DE SELEÇÃO ---
print("\nQual região você deseja validar?")
for key, val in REGIOES.items():
    print(f"[{key}] - {val['nome']}")

while True:
    escolha = input("Digite o número da região (1 a 5): ").strip()
    if escolha in REGIOES:
        regiao = REGIOES[escolha]
        break
    print("ERRO: Escolha uma opção válida.")

try:
    ano_inicial = int(input(f"Validar a partir de qual ano? (Ex: 2015): ").strip())
except ValueError:
    ano_inicial = 2015

# --- 2. OBTER A FONTE DA VERDADE (IBGE) ---
nome_regiao = regiao['nome']
print(f"\nBaixando a lista oficial de municípios da região {nome_regiao} no IBGE...")

url_ibge = f"https://servicodados.ibge.gov.br/api/v1/localidades/regioes/{regiao['id_ibge']}/municipios"
cidades_regiao = requests.get(url_ibge).json()
total_cidades = len(cidades_regiao)

print(f"✅ {total_cidades} municípios carregados. Iniciando auditoria...\n")

# --- 3. FASE DE AUDITORIA ---
arquivos_faltantes = [] # Vai guardar o que precisa ser baixado

for ano in range(ano_inicial, ano_final + 1):
    faltam_neste_ano = 0
    
    for cidade in cidades_regiao:
        cod_ibge = str(cidade['id'])
        nome_cidade = cidade['nome'].replace("/", "-").replace("\\", "-")
        sigla_uf = cidade['microrregiao']['mesorregiao']['UF']['sigla']
        
        # Reconstrói o caminho exato onde o arquivo DEVERIA estar
        nome_arquivo = f"{cod_ibge}_{nome_cidade}_{sigla_uf}.csv"
        caminho_esperado = os.path.join(base_dir, nome_regiao, sigla_uf, str(ano), nome_arquivo)
        
        # Verifica se o arquivo existe fisicamente no HD
        if not os.path.exists(caminho_esperado):
            arquivos_faltantes.append({
                'ano': ano,
                'cod_ibge': cod_ibge,
                'nome_cidade': nome_cidade,
                'sigla_uf': sigla_uf,
                'caminho': caminho_esperado
            })
            faltam_neste_ano += 1
            
    print(f" 📅 Ano {ano}: {total_cidades - faltam_neste_ano}/{total_cidades} arquivos presentes. " +
          (f"❌ Faltam {faltam_neste_ano}" if faltam_neste_ano > 0 else "✅ Tudo OK!"))

# --- 4. FASE DE RESGATE (DOWNLOAD CORRETIVO) ---
total_faltantes = len(arquivos_faltantes)

if total_faltantes == 0:
    print("\n🎉 AUDITORIA CONCLUÍDA: Sua base de dados está 100% íntegra! Não há o que corrigir.")
    exit()

print("\n" + "!"*60)
print(f" ⚠️ ALERTA: Foram detectados {total_faltantes} arquivos ausentes.")
print("!"*60)

resposta = input("\nDeseja iniciar o resgate (download) desses arquivos agora? (S/n): ").strip().lower()

if resposta == 'n':
    print("Operação cancelada. A base continua incompleta.")
    exit()

print("\n🚀 INICIANDO DOWNLOAD DE RESGATE...")
base_url = "https://info.dengue.mat.br/api/alertcity"
sucessos = 0
erros = 0
vazios = 0

for i, item in enumerate(arquivos_faltantes):
    ano = item['ano']
    cod_ibge = item['cod_ibge']
    sigla_uf = item['sigla_uf']
    nome_cidade = item['nome_cidade']
    caminho = item['caminho']
    
    # Garante que a pasta existe (caso o ano inteiro tenha falhado)
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    
    print(f"\rResgatando [{i+1}/{total_faltantes}]: {nome_cidade}-{sigla_uf} ({ano})...", end="")
    
    params = {
        'geocode': cod_ibge,
        'disease': 'dengue',
        'format': 'json',
        'ew_start': 1,
        'ey_start': ano,
        'ew_end': 53,
        'ey_end': ano
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                df = pd.DataFrame(data)
                df.to_csv(caminho, index=False)
                sucessos += 1
            else:
                # O InfoDengue realmente não tem dados dessa cidade/ano
                vazios += 1 
        else:
            erros += 1
            
    except Exception as e:
        erros += 1
        
    time.sleep(0.2) # Pausa um pouco maior no resgate para garantir estabilidade

print("\n\n" + "="*60)
print(" 🛠️ OPERAÇÃO DE RESGATE FINALIZADA!")
print(f" 📊 Resumo do Resgate: Salvos: {sucessos} | InfoDengue sem dados: {vazios} | Erros: {erros}")
print(" Recomendação: Rode este script novamente para confirmar se restaram erros.")
print("="*60)