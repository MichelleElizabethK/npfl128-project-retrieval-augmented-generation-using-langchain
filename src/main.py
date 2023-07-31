import argparse

from document_loader import DocumentLoader
from vector_stores_and_embeddings import VectorStore
from retrieval_chain import RetrievalChain, ConversationalRetrievalQAChain

parser = argparse.ArgumentParser(description="Command Line Arguments for the Retrival Augmented Generation System",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--type", help="type of the source", default="pdf", choices=["pdf", "youtube", "url"], type=str, required=True)
parser.add_argument("--source", help="Source of the information. For youtube videos, provide only the id. For others, provide the full path.", type=str, required=True)
parser.add_argument("--mode", help="Either chat with the data or simply retrieve information", default="chat", choices=["chat", "retrieve"], type=str)


def main():
    args = parser.parse_args()
    config = vars(args)
    # Load the source of data
    docs = DocumentLoader(config["type"], config['source'])
    # Split, embed and index the documents
    vector_store = VectorStore(docs.split_docs)

    # initialise the required QA chain with the index
    if config["mode"]=="chat":
        qa_chain = ConversationalRetrievalQAChain(vector_store.vectordb)
    else:
        qa_chain = RetrievalChain(vector_store.vectordb)
    
    print("The data you provided has been loaded. Now you can chat with your data! \n Type a question and press enter to get started. To exit, type exit.")

    while True:
        question = input()
        if question.lower() == 'exit':
            print('Goodbye!!')
            break
        answer = qa_chain.get_response(question)
        print(answer["response"])


if __name__ == '__main__':
    main()
