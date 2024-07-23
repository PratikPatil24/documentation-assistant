import os

from dotenv import load_dotenv
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from typing import List, Any, Dict
from langchain.prompts import PromptTemplate

load_dotenv()


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(
        index_name=os.environ["PINECONE_INDEX_NAME"], embedding=embeddings_model
    )
    llm = ChatOpenAI(model="gpt-4o", verbose=True)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    # rephrase_prompt_template = """
    # Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
    #
    # Chat History:
    #
    # {chat_history}
    #
    # Follow Up Input: {input}
    #
    # Standalone Question:"""
    # print(rephrase_prompt_template)
    # rephrase_prompt = PromptTemplate(
    #     template=rephrase_prompt_template, input_variables=["input", "chat_history"]
    # )
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=vectorstore.as_retriever(), prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    return result


if __name__ == "__main__":
    print("Starting Retrieval QA")
    response = run_llm(query="What is Langchain Chain?")
    print(response["answer"])
