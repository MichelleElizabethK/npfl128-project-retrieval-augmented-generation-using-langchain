from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os

model_path = os.environ["MODEL_PATH"]

class RetrievalChain:

    llm = GPT4All(model=model_path, max_tokens=2048)
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
                {context}
                Question: {question}
                Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    def __init__(self, vectordb) -> None:
        
        self.qa_chain = RetrievalQA.from_chain_type(
            self.llm,
            retriever=vectordb.as_retriever(),
                return_source_documents=True,
            chain_type_kwargs={"prompt": self.QA_CHAIN_PROMPT},
            memory=self.memory
        )
        self.qa = ConversationalRetrievalChain.from_llm(
                self.llm,
                retriever=vectordb.as_retriever(),
                chain_type="stuff",
                memory=self.memory
            )

    def get_result(self, question: str) -> dict:
        result = self.qa_chain({"query": question})
        return {
            "result": result["result"],
            "source": result["source_documents"]
        }
    
    def chat(self, question: str) -> dict:
        result = self.qa({"question": question})
        return result["answer"]