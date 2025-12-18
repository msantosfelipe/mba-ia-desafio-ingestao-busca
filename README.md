# Desafio MBA Engenharia de Software com IA - Full Cycle

Descreva abaixo como executar a sua solução.

# Preparando o ambiente:
## Criar venv
- Executar `python3 -m venv venv`
- Executar `source venv/bin/activate`

## Instalar dependencias
- Executar `pip install -r requirements.txt`

## Criar arquivo .env
- Executar `cp .env.example .env`
- Preencher variáveis GOOGLE_API_KEY e OPENAI_API_KEY, as outras devem ser alteradas de acordo com preferência de execução.
    - Foram adicionadas as variáveis USE_OLLAMA, OLLAMA_BASE_URL e OLLAMA_MODEL_NAME, para execução do embedding localmente no caso de problemas de quota na OPENAI

## Iniciando docker
- Executar `docker compose up -d`


# Executando o projeto
## Inserindo dados do PDF
- Executar `python src/ingest.py`
    - Lê o pdf `document.pdf`, separa em chunks e salva vetorizado no banco
