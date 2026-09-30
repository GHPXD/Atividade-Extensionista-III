"""
PIPELINE DENGO - Execução Completa
===============================
Script principal para executar todo o pipeline de treinamento.

Uso:
    python main.py --all          # Executa todas as etapas
    python main.py --prep         # Apenas preparação de dados
    python main.py --seq          # Apenas criação de sequências
    python main.py --train        # Apenas treinamento
    python main.py --eval         # Apenas avaliação
"""

import subprocess
import sys
import argparse
import os
from datetime import datetime

# Cores para terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.FAIL}❌ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.WARNING}⚠️ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ️ {text}{Colors.END}")

def run_script(script_name, description):
    """Executa um script Python e retorna se teve sucesso"""
    print_info(f"Executando: {script_name}")
    print(f"   {description}")
    print()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False
        )
        print_success(f"{script_name} concluído com sucesso!")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Erro ao executar {script_name}")
        return False
    except FileNotFoundError:
        print_error(f"Arquivo não encontrado: {script_name}")
        return False

def check_dependencies():
    """Verifica se as dependências estão instaladas"""
    print_header("🔍 VERIFICANDO DEPENDÊNCIAS")
    
    deps = [
        ('pandas', 'Manipulação de dados'),
        ('numpy', 'Computação numérica'),
        ('sklearn', 'Machine Learning'),
        ('tensorflow', 'Deep Learning'),
        ('matplotlib', 'Visualização'),
        ('joblib', 'Serialização')
    ]
    
    all_ok = True
    for module, desc in deps:
        try:
            __import__(module)
            print(f"   ✅ {module}: OK")
        except ImportError:
            print(f"   ❌ {module}: NÃO ENCONTRADO ({desc})")
            all_ok = False
    
    if not all_ok:
        print()
        print_warning("Instale as dependências faltantes:")
        print("   pip install pandas numpy scikit-learn tensorflow matplotlib joblib")
        return False
    
    print()
    print_success("Todas as dependências estão instaladas!")
    return True

def check_data_files():
    """Verifica se os arquivos de dados existem"""
    print_header("📁 VERIFICANDO ARQUIVOS DE DADOS")
    
    required_dirs = ['dados_dengue']
    required_files = []
    
    # Verificar diretório de dados
    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            n_files = len([f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))])
            print(f"   ✅ {dir_path}/: {n_files} subpastas (anos)")
        else:
            print(f"   ❌ {dir_path}/: NÃO ENCONTRADO")
            print_warning("Execute o download dos dados do InfoDengue primeiro!")
            return False
    
    print()
    print_success("Arquivos de dados encontrados!")
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Pipeline principal para treinamento do modelo Dengo AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
    python main.py --all          # Pipeline completo
    python main.py --prep --seq   # Preparação + Sequências
    python main.py --train --eval # Treino + Avaliação
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Executa todas as etapas')
    parser.add_argument('--prep', action='store_true', help='Preparar dataset')
    parser.add_argument('--seq', action='store_true', help='Criar sequências')
    parser.add_argument('--train', action='store_true', help='Treinar modelo')
    parser.add_argument('--eval', action='store_true', help='Avaliar modelo')
    parser.add_argument('--check', action='store_true', help='Apenas verificar dependências')
    
    args = parser.parse_args()
    
    # Se nenhum argumento, mostrar help
    if not any([args.all, args.prep, args.seq, args.train, args.eval, args.check]):
        parser.print_help()
        return
    
    print_header("🚀 DENGO AI - PIPELINE PRINCIPAL")
    print(f"   Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    print(f"   Versão: Principal")
    
    # Verificar dependências
    if not check_dependencies():
        return
    
    if args.check:
        check_data_files()
        return
    
    # Verificar dados
    if args.all or args.prep:
        if not check_data_files():
            return
    
    # Definir etapas a executar
    etapas = []
    
    if args.all:
        etapas = [
            ('preparar_dataset.py', 'Consolidar e limpar dados de todas as cidades'),
            ('criar_sequencias.py', 'Criar janelas temporais e normalizar por cidade'),
            ('treinar_modelo.py', 'Treinar modelo LSTM com embedding e attention'),
            ('testar_ia.py', 'Avaliar modelo com métricas completas')
        ]
    else:
        if args.prep:
            etapas.append(('preparar_dataset.py', 'Consolidar e limpar dados'))
        if args.seq:
            etapas.append(('criar_sequencias.py', 'Criar janelas temporais'))
        if args.train:
            etapas.append(('treinar_modelo.py', 'Treinar modelo'))
        if args.eval:
            etapas.append(('testar_ia.py', 'Avaliar modelo'))
    
    # Executar etapas
    total_etapas = len(etapas)
    etapas_ok = 0
    
    for i, (script, desc) in enumerate(etapas, 1):
        print_header(f"📌 ETAPA {i}/{total_etapas}: {script}")
        
        if run_script(script, desc):
            etapas_ok += 1
        else:
            print_error(f"Pipeline interrompido na etapa {i}")
            break
        
        print()
    
    # Resumo final
    print_header("📋 RESUMO DA EXECUÇÃO")
    
    print(f"   Etapas executadas: {etapas_ok}/{total_etapas}")
    
    if etapas_ok == total_etapas:
        print_success("Pipeline concluído com sucesso!")
        print()
        print("   📁 Arquivos gerados:")
        
        arquivos_gerados = [
            'DATASET_PARA_IA.csv',
            'cidades_por_porte.json',
            'X_train.npy', 'y_train.npy',
            'X_test.npy', 'y_test.npy',
            'scalers_por_cidade.pkl',
            'dengo_ai.keras',
            'grafico_treinamento.png',
            'avaliacao_modelo.png',
            'relatorio_avaliacao.json'
        ]
        
        for arq in arquivos_gerados:
            if os.path.exists(arq):
                size = os.path.getsize(arq)
                if size > 1024*1024:
                    size_str = f"{size/1024/1024:.1f} MB"
                elif size > 1024:
                    size_str = f"{size/1024:.1f} KB"
                else:
                    size_str = f"{size} bytes"
                print(f"      ✅ {arq} ({size_str})")
        
        print()
        print("   💡 Próximos passos:")
        print("      1. Verifique o R² no relatório de avaliação")
        print("      2. Se R² < 0.5, considere treinar modelos por porte de cidade")
        print("      3. Copie dengo_ai.keras para o backend do app")
    else:
        print_error("Pipeline não foi concluído completamente.")
        print()
        print("   💡 Verifique os erros acima e tente novamente.")

if __name__ == '__main__':
    main()