import json
import sys
import os


def validar_no(no, caminho="raiz"):
    """
    Valida recursivamente a estrutura de um nó da árvore.
    
    Args:
        no: Nó atual da árvore
        caminho: Caminho do nó na árvore (para mensagens de erro)
    
    Raises:
        ValueError: Se o nó estiver mal formado
    """
    # Verifica se é um dicionário
    if not isinstance(no, dict):
        raise ValueError(f"Nó em '{caminho}' não é um dicionário válido")
    
    # Se tem 'result', é um nó folha
    if 'result' in no:
        return
    
    # Se não tem 'result', deve ter 'question', 'yes' e 'no'
    if 'question' not in no:
        raise ValueError(f"Nó em '{caminho}' não tem 'question' nem 'result'")
    
    if not isinstance(no['question'], str) or not no['question'].strip():
        raise ValueError(f"Nó em '{caminho}' tem 'question' inválida")
    
    if 'yes' not in no:
        raise ValueError(f"Nó em '{caminho}' não tem opção 'yes'")
    
    if 'no' not in no:
        raise ValueError(f"Nó em '{caminho}' não tem opção 'no'")
    
    # Valida recursivamente os nós filhos
    validar_no(no['yes'], f"{caminho} -> yes")
    validar_no(no['no'], f"{caminho} -> no")


def carregar_arvore(arquivo='ia-trabalho-2025-2/src/part1_tree_manual/perguntas.json'):
    """
    Carrega e valida a árvore de decisão do arquivo JSON.
    
    Args:
        arquivo: Caminho do arquivo JSON
    
    Returns:
        Dicionário com a árvore de decisão
    
    Raises:
        FileNotFoundError: Se o arquivo não existir
        json.JSONDecodeError: Se o arquivo não for um JSON válido
        ValueError: Se a estrutura da árvore estiver incorreta
    """
    try:
        # Verifica se o arquivo existe
        if not os.path.exists(arquivo):
            raise FileNotFoundError(f"Arquivo '{arquivo}' não encontrado")
        
        # Tenta abrir e ler o arquivo
        with open(arquivo, 'r', encoding='utf-8') as f:
            arvore = json.load(f)
        
        # Valida a estrutura da árvore
        validar_no(arvore)
        
        return arvore
    
    except FileNotFoundError as e:
        print(f"\n❌ ERRO: {e}")
        print("Certifique-se de que o arquivo 'perguntas.json' existe no diretório.")
        sys.exit(1)
    
    except json.JSONDecodeError as e:
        print(f"\n❌ ERRO: Arquivo JSON inválido")
        print(f"Detalhes: {e}")
        sys.exit(1)
    
    except ValueError as e:
        print(f"\n❌ ERRO: Estrutura da árvore inválida")
        print(f"Detalhes: {e}")
        sys.exit(1)
    
    except PermissionError:
        print(f"\n❌ ERRO: Sem permissão para ler o arquivo '{arquivo}'")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ ERRO inesperado ao carregar o arquivo:")
        print(f"Detalhes: {e}")
        sys.exit(1)


def obter_resposta(pergunta):
    """
    Obtém uma resposta sim/não do usuário.
    
    Args:
        pergunta: String com a pergunta a ser feita
    
    Returns:
        'yes' ou 'no'
    """
    while True:
        try:
            resposta = input(f"{pergunta} (s/n): ").strip().lower()
            if resposta in ['s', 'sim', 'yes', 'y']:
                return 'yes'
            elif resposta in ['n', 'não', 'nao', 'no']:
                return 'no'
            else:
                print("Por favor, responda com 's' para sim ou 'n' para não.")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Programa interrompido pelo usuário.")
            sys.exit(0)
        
        except EOFError:
            print("\n\n⚠️  Entrada inválida. Encerrando programa.")
            sys.exit(0)


def navegar_arvore(no):
    """
    Navega recursivamente pela árvore de decisão.
    
    Args:
        no: Nó atual da árvore
    
    Returns:
        Lista de hobbies recomendados
    """
    try:
        # Verifica se chegou em um resultado final
        if 'result' in no:
            return no['result']
        
        # Faz a pergunta e obtém a resposta
        pergunta = no['question']
        resposta = obter_resposta(pergunta)
        
        # Navega para o próximo nó baseado na resposta
        proximo_no = no[resposta]
        return navegar_arvore(proximo_no)
    
    except KeyError as e:
        print(f"\n❌ ERRO: Estrutura da árvore corrompida")
        print(f"Chave ausente: {e}")
        sys.exit(1)
    
    except TypeError as e:
        print(f"\n❌ ERRO: Tipo de dado inválido na árvore")
        print(f"Detalhes: {e}")
        sys.exit(1)


def main():
    """Função principal que executa o sistema de recomendação de hobbies."""
    print("=" * 60)
    print("SISTEMA DE RECOMENDAÇÃO DE HOBBIES")
    print("=" * 60)
    print("\nResponda às perguntas a seguir para descobrir qual hobby")
    print("combina melhor com você!\n")
    
    # Carrega a árvore de decisão (com validação)
    arvore = carregar_arvore()
    
    # Navega pela árvore e obtém o resultado
    hobbies_recomendados = navegar_arvore(arvore)
    
    # Exibe os resultados
    print("\n" + "=" * 60)
    print("HOBBIES RECOMENDADOS PARA VOCÊ:")
    print("=" * 60)
    for i, hobby in enumerate(hobbies_recomendados, 1):
        print(f"{i}. {hobby}")
    print("=" * 60)
    print("\nDivirta-se! 🎉")


if __name__ == "__main__":
    main()
