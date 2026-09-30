import pandas as pd
import requests
import os
import time

# --- CONFIGURAÇÕES GOVTECH ---
ano_final = 2025  # Limite fixo
base_dir = "dados_dengue_brasil"

REGIOES = {
    1: {"nome": "Norte", "id_ibge": 1},
    2: {"nome": "Nordeste", "id_ibge": 2},
    3: {"nome": "Sudeste", "id_ibge": 3},
    4: {"nome": "Sul", "id_ibge": 4},
    5: {"nome": "Centro-Oeste", "id_ibge": 5}
}

# 💡 CORREÇÃO: Função blindada para extrair a UF, lidando com anomalias do IBGE (ex: Distrito Federal)
def extrair_uf(cidade_json):
    try:
        # Padrão tradicional
        if cidade_json.get('microrregiao') is not None:
            return cidade_json['microrregiao']['mesorregiao']['UF']['sigla']
        # Novo padrão do IBGE (pós-2017)
        elif cidade_json.get('regiao-imediata') is not None:
            return cidade_json['regiao-imediata']['regiao-intermediaria']['UF']['sigla']
        # Fallback para o Distrito Federal
        elif "Brasília" in cidade_json.get('nome', ''):
            return "DF"
        else:
            return "CO" # Fallback genérico para Centro-Oeste
    except Exception:
        return "XX"

print("#" * 70)
print(" 🇧🇷 SISTEMA DE DOWNLOAD - INFODENGUE (MODO BLINDADO v2)")
print("#" * 70)

print("\nEscolha qual região você deseja baixar os dados:")
for key, val in REGIOES.items():
    print(f"[{key}] - {val['nome']}")
print("[6] - TODAS AS REGIÕES em sequência")

regioes_para_baixar = []

while True:
    escolha = input("Digite sua opção (1 a 6): ").strip()
    
    if escolha.isdigit() and 1 <= int(escolha) <= 5:
        regioes_para_baixar.append(REGIOES[int(escolha)])
        print(f"--> Região {REGIOES[int(escolha)]['nome']} selecionada!")
        break
    elif escolha == '6':
        print("\nVocê escolheu TODAS AS REGIÕES.")
        inicio_todas = input("Deseja começar a partir de qual delas? (1 a 5) [Padrão=1]: ").strip()
        inicio_idx = int(inicio_todas) if inicio_todas.isdigit() and 1 <= int(inicio_todas) <= 5 else 1
        
        ordem_regioes = list(range(inicio_idx, 6)) + list(range(1, inicio_idx))
        for i in ordem_regioes:
            regioes_para_baixar.append(REGIOES[i])
            
        nomes_fila = [r['nome'] for r in regioes_para_baixar]
        print(f"--> Fila de download criada: {', '.join(nomes_fila)}")
        break
    else:
        print("ERRO: Escolha uma opção válida de 1 a 6.")

while True:
    try:
        entrada = input(f"\nEm qual ano deseja começar? (Entre 2015 e {ano_final}) [Padrão=2015]: ").strip()
        ano_inicial = int(entrada) if entrada else 2015
        
        if 2015 <= ano_inicial <= ano_final:
            break
        else:
            print(f"ERRO: O ano deve estar entre 2015 e {ano_final}.")
    except ValueError:
        print("ERRO: Digite um número válido.")

modo_auto = input("\nDeseja rodar no MODO AUTOMÁTICO (sem pausas entre os anos/regiões)? (S/n): ").strip().lower()
modo_automatico = False if modo_auto == 'n' else True

base_url = "https://info.dengue.mat.br/api/alertcity"

for regiao_atual in regioes_para_baixar:
    nome_regiao = regiao_atual['nome']
    id_regiao_ibge = regiao_atual['id_ibge']

    print(f"\n" + "*"*70)
    print(f" 🌍 INICIANDO OPERAÇÕES NA REGIÃO: {nome_regiao.upper()}")
    print("*"*70)

    try:
        url_ibge = f"https://servicodados.ibge.gov.br/api/v1/localidades/regioes/{id_regiao_ibge}/municipios"
        cidades_brutas = requests.get(url_ibge).json()
        
        # 💡 Aqui a nova função "extrair_uf" é usada, evitando o crash!
        cidades_regiao = sorted(
            cidades_brutas, 
            key=lambda x: (extrair_uf(x), x.get('nome', ''))
        )
        total_cidades = len(cidades_regiao)
        
        total_por_estado = {}
        for c in cidades_regiao:
            uf = extrair_uf(c)
            total_por_estado[uf] = total_por_estado.get(uf, 0) + 1
            
    except Exception as e:
        print(f"Erro fatal ao conectar com IBGE na região {nome_regiao}: {e}")
        continue 

    for ano in range(ano_inicial, ano_final + 1):
        print(f"\n" + "="*70)
        print(f" 📥 BAIXANDO DADOS DE: {ano} | REGIÃO: {nome_regiao}")
        print(f"="*70)
        
        sucessos = 0
        erros = 0
        vazios = 0
        ignorados = 0 
        
        progresso_estado = {uf: 0 for uf in total_por_estado.keys()}
        
        for i, cidade in enumerate(cidades_regiao):
            cod_ibge = str(cidade['id'])
            nome_cidade = cidade.get('nome', 'Desconhecida').replace("/", "-").replace("\\", "-")
            sigla_uf = extrair_uf(cidade)
            
            progresso_estado[sigla_uf] += 1
            atual_estado = progresso_estado[sigla_uf]
            total_estado = total_por_estado[sigla_uf]
            
            dir_destino = os.path.join(base_dir, nome_regiao, sigla_uf, str(ano))
            os.makedirs(dir_destino, exist_ok=True)
            
            nome_arquivo = f"{cod_ibge}_{nome_cidade}_{sigla_uf}.csv"
            caminho_arquivo = os.path.join(dir_destino, nome_arquivo)
            
            if os.path.exists(caminho_arquivo):
                ignorados += 1
                linha_progresso = f"\r⏳ [Ano: {ano}] Geral: [{i+1}/{total_cidades}] | Estado {sigla_uf}: [{atual_estado}/{total_estado}]"
                linha_cidade = f"\n\r   ⏩ Pulando (Já existe): {nome_cidade}...{' ' * 20}"
                print(f"{linha_progresso}{linha_cidade}\033[F", end="", flush=True)
                continue 
            
            linha_progresso = f"\r⏳ [Ano: {ano}] Geral: [{i+1}/{total_cidades}] | Estado {sigla_uf}: [{atual_estado}/{total_estado}]"
            linha_cidade = f"\n\r   -> Baixando: {nome_cidade}...{' ' * 30}"
            print(f"{linha_progresso}{linha_cidade}\033[F", end="", flush=True)
            
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
                response = requests.get(base_url, params=params, timeout=20)
                
                if response.status_code == 200:
                    data = response.json()
                    if data and len(data) > 0:
                        df = pd.DataFrame(data)
                        df.to_csv(caminho_arquivo, index=False)
                        sucessos += 1
                    else:
                        vazios += 1 
                else:
                    erros += 1
            except Exception as e:
                erros += 1
                
            time.sleep(0.1) 
                
        print("\n\n") 
        print(f"--> Ano {ano} finalizado para a região {nome_regiao}!")
        print(f"📊 Resumo: Salvos: {sucessos} | Já existiam (Pulados): {ignorados} | Sem dados: {vazios} | Erros: {erros}")
        
        if not modo_automatico and ano < ano_final:
            while True:
                resposta = input(f"\nDeseja iniciar o download do próximo ano ({ano + 1})? (S/n): ").strip().lower()
                if resposta == '' or resposta == 's':
                    break 
                elif resposta == 'n':
                    print("Processo interrompido pelo usuário. Até logo!")
                    exit()
                    
    ano_inicial = 2015

print("\n" + "#"*70)
print(f" 🚀 SUCESSO ABSOLUTO! TODAS AS OPERAÇÕES FORAM CONCLUÍDAS.")
print("#"*70)