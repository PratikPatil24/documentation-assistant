import os

from dotenv import load_dotenv
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()


def insert_docs():
    loader = ReadTheDocsLoader(
        "/home/pratik-patil/Learnings/Langchain/documentation-assistant/langchain-docs/api.python.langchain.com/en/latest"
    )
    raw_documents = loader.load()
    print(f"Number of docs loaded {len(raw_documents)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} docs to vector db")

    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

    PineconeVectorStore.from_documents(
        documents,
        embeddings_model,
        index_name=os.environ["PINECONE_INDEX_NAME"],
    )
    print("Data Insertion Complete!")


if __name__ == "__main__":
    print("Inserting Data...")
    insert_docs()
