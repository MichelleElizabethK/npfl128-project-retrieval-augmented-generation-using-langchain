from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.embeddings import OpenAIEmbeddings

import chromadb
import os
import openai


class VectorStore:

    openai.api_key = os.environ["OPENAI_API_KEY"]

    embedding = OpenAIEmbeddings()
    llm = OpenAI(temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    persist_directory = './chromadb/'
    chroma_client = chromadb.EphemeralClient()

    def __init__(self, splits) -> None:
        """Create the index from the split documents

        Args:
            splits : The main document split into chunks
        """

        # initialise the vector db with the given splits and embedding function
        self.vectordb = Chroma.from_documents(
            documents=splits,
            embedding=self.embedding,
            client=self.chroma_client,
        )

        # initialise a retriever based on contextual compression
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.vectordb.as_retriever(search_type = "mmr")
        )

    def retrieve_relevant_docs(self, question: str) -> list:
        """Use the compression retriever to extract documents relevant to the query

        Args:
            question (str): the query for the search

        Returns:
            The list of relevant documents
        """
        return self.compression_retriever.get_relevant_documents(question)
    
    def retrieve_docs_similarity_search(self, question: str) -> list:
        """Use the similarity search to search for documents from the vector db
           based on the question

        Args:
            question (str): the query for the search

        Returns:
            The list of relevant documents
        """
        return self.vectordb.similarity_search(question)
