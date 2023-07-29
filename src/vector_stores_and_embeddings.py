from langchain.vectorstores import Chroma
from langchain.llms import GPT4All
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.embeddings import HuggingFaceEmbeddings

import chromadb
import os

model_path = os.environ["MODEL_PATH"]

class VectorStore:
    embedding = HuggingFaceEmbeddings()
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    llm = GPT4All(model=model_path, max_tokens=2048)
    compressor = LLMChainExtractor.from_llm(llm)
    persist_directory = './chromadb/'
    chroma_client = chromadb.EphemeralClient()

    def __init__(self, splits) -> None:
        self.vectordb = Chroma.from_documents(
            documents=splits,
            embedding=self.embedding,
            client=self.chroma_client,
        )

        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.vectordb.as_retriever(search_type = "mmr")
        )

    def retrieve_relevant_docs(self, question) -> None:
        return self.compression_retriever.get_relevant_documents(question)
    
    def retrieve_docs_similarity_search(self, question) -> str:
        return self.vectordb.similarity_search(question)