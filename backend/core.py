import os

from dotenv import load_dotenv
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

load_dotenv()


def run_llm(query: str):
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings_model
    )
    retrieval_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    print(retrieval_prompt)

    llm = ChatOpenAI(model="gpt-4o", verbose=True)

    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    result = retrieval_chain.invoke({"input": query})
    return result


if __name__ == "__main__":
    print("Starting Retrieval QA")
    response = run_llm(query="What is Langchain Chain?")
    print(response["answer"])
