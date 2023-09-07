from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

class RetrievalChain:

    #  create a custom template to control how the LLM responds to queries
    template = """Use the following pieces of context to answer the question at the end.
                If you don't know the answer say "Sorry, I am not sure I understand." 
                If question is out of context, say "Sorry, this does not seem to be relevant to the data you uploaded".
                Don't try to make up an answer. Keep the answer as concise as possible.
                {context}
                Question: {question}
                Helpful Answer:"""
    prompt = PromptTemplate.from_template(template)

    def __init__(self, vectordb) -> None:
        """Initialises the RetrievalQA chain with the vector db as the retriever
           and the custom prompt 

        Args:
            vectordb : the index created from the documents
        """
        self.retrieval_qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt},
        )


    def get_response(self, query: str) -> dict:
        """Use the question answering chain to feed in questions and get a response

        Args:
            query (str): the user query

        Returns:
            dict: a dictionary containing the response and the source document
        """
        result = self.retrieval_qa_chain({"query": query})
        return {
            "response": result["result"],
            "source": result["source_documents"]
        }
    

    
class ConversationalRetrievalQAChain:

    # initialise a conversation buffer memory to keep track of the chat
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
        )

    def __init__(self, vectordb) -> None:
        """Initialises the RetrievalQA chain with the vector db as the retriever
           and the custom prompt 

        Args:
            vectordb : the index created from the documents
        """

        template = (
            """Use the following pieces of context to answer the question at the end.
                If you don't know the answer say something which means "Sorry, I am not sure I understand." 
                If question is out of context, say something which means "Sorry, this does not seem to be relevant to the data you uploaded".
                Don't try to make up an answer. Keep the answer as concise as possible. Be polite and observe general pleasantries.
                {context}
                Question: {question}
                Helpful Answer:"""
        )
        prompt = PromptTemplate.from_template(template)

        # initialise the conversational retrieval chain with memory with custom prompt
        self.conv_qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vectordb.as_retriever(), memory=self.memory, 
            return_source_documents=True, combine_docs_chain_kwargs={'prompt': prompt}
        )

    
    def get_response(self, question: str) -> dict:
        """Use the question answering chain to feed in questions and get a response

        Args:
            question (str): the user's message

        Returns:
            dict: a dictionary containing the response and the source document
        """
        result = self.conv_qa_chain({"question": question})
        return {
            "response": result["answer"],
            "source": result["source_documents"]
        }
