import json

def carregar_arvore(arquivo='perguntas.json'):
    # Carrega a árvore de decisão do arquivo JSON.
    with open(arquivo, 'r', encoding='utf-8') as f:
        return json.load(f)


def obter_resposta(pergunta):
    # Obtém uma resposta sim/não do usuário.
    while True:
        resposta = input(f"{pergunta} (s/n): ").strip().lower()
        if resposta in ['s', 'sim', 'yes', 'y']:
            return 'yes'
        elif resposta in ['n', 'não', 'nao', 'no']:
            return 'no'
        else:
            print("Por favor, responda com 's' para sim ou 'n' para não.")


def navegar_arvore(no):
    # Navega recursivamente pela árvore de decisão.
    
    # Verifica se chegou em um resultado final
    if 'result' in no:
        return no['result']
    
    # Faz a pergunta e obtém a resposta
    pergunta = no['question']
    resposta = obter_resposta(pergunta)
    
    # Navega para o próximo nó baseado na resposta
    proximo_no = no[resposta]
    return navegar_arvore(proximo_no)


def main():
    
    print("=" * 60)
    print("SISTEMA DE RECOMENDAÇÃO DE HOBBIES")
    print("=" * 60)
    print("\nResponda às perguntas a seguir para descobrir qual hobby")
    print("combina melhor com você!\n")
    
    # Carrega a árvore de decisão
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
