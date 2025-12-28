import os
from dotenv import load_dotenv

from langchain_postgres import PGVector
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain_core.runnables import chain


load_dotenv()


USE_OLLAMA = os.getenv("USE_OLLAMA", "False").lower() == "true"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
DATABASE_URL = os.getenv("DATABASE_URL")
PG_VECTOR_COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME")


PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""


@chain
def _get_context(question_dict: dict) -> str:
    question = question_dict["pergunta"]
    if USE_OLLAMA:
        embeddings = OllamaEmbeddings(
            model=OLLAMA_MODEL_NAME,
            base_url=OLLAMA_BASE_URL
        )
    else:
        embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

    store = PGVector(
        embeddings=embeddings,
        collection_name=PG_VECTOR_COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )

    results = store.similarity_search_with_score(question, k=10)

    context = "\n".join([doc.page_content for doc, _ in results])
    return {"contexto": context, "pergunta": question}


def search_prompt(question=None):
    if question is None:
        return

    question_template = PromptTemplate(
      input_variables=["contexto", "pergunta"],
      template=PROMPT_TEMPLATE,
    )

    gemini = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

    chain = _get_context | question_template | gemini

    response = chain.invoke({"pergunta": question})
    return response.content
