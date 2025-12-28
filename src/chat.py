# Perguntas de exemplo:
# Qual o faturamento da Empresa SuperTechIABrazil?
# Qual o faturamento da Empresa Brava Biotech Holding?

# Pergunta fora de contexto:
# Quantos clientes temos em 2024?

from search import search_prompt

def main():
    print('🤖: Bem vindo ao ChatBot do Faturamento! Faça sua pergunta. Digite "sair" para encerrar.')
    while True:
        question = input("👤: ")

        if question.lower() == "sair":
            print("🤖: Até mais!")
            break

        chain_response = search_prompt(question)

        if not chain_response:
            print("🤖: Não foi possível iniciar o chat. Verifique os erros de inicialização.")
            return

        print(f"🤖: {chain_response}")

if __name__ == "__main__":
    main()
