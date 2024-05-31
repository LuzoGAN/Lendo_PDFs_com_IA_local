import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain.schema.document import Document
from langchain.vectorstores.chroma import Chroma

CHROMA_PATH = "chroma"
DATA_PATH = r"C:\Users\performance\Downloads\data"

# Divindo os blocos
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

def split_documents(documents: list[Document]):
  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=800,
      chunk_overlap=80,
      length_function=len,
      is_separator_regex=False
  )
  return text_splitter.split_documents(documents)

from langchain_community.embeddings.ollama import OllamaEmbeddings

def get_embedding_function():
  embeddings = OllamaEmbeddings(model='llama3')
  return embeddings

# Criando o banco de dados com as informações
#from get_embedding_function import get_embedding_function
from langchain.vectorstores.chroma import Chroma

def add_to_chroma(chunks: list[Document]):
  db = Chroma(
      persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
  )
  db.add_documents(new_chunks, ids=new_chunk_ids)
  db.persist()

from langchain.vectorstores.chroma import Chroma

db = Chroma(
    persist_directory=CHROMA_PATH,
)

def query_rag(query_text: str):
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)

    print(results)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(model="llama3")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]

    print(sources)

    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

PROMPT_TEMPLATE = """
Responda à pergunta com base apenas no seguinte contexto:

{context}

---

Responda à pergunta com base no contexto acima: {question}
"""

def main():
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: list[Document]):
    # Carregando databse que já existe
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculando os IDS da pagina
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Adicionando ou atualizando os dados
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Numeros de documentos no DB: {len(existing_ids)}")

    # Adicionando somente documents que não tenham no banco de dados
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adicionando documentos: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        db.persist()
    else:
        print("✅ Sem documentos para adicionar")

  def calculate_chunk_ids(chunks):

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # se a página for a mesma aumenta no index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # calculando chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # adicionando ao metadata
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()

query_rag('Em qual laboratório foi feitos os testes?')
