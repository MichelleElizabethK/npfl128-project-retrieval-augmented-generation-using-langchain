import argparse

from document_loader import DocumentLoader
from vector_stores_and_embeddings import VectorStore
from retrieval_chain import RetrievalChain

parser = argparse.ArgumentParser(description="Command Line Arguments for the Retrival Augmented Generation System",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--type", help="type of the source", default="pdf", choices=["pdf", "youtube", "url"], type=str)
parser.add_argument("--source", help="Source of the information. For youtube videos, provide only the id. For others, provide the full path.", type=str)


def main():
    args = parser.parse_args()
    config = vars(args)
    docs = DocumentLoader(config["type"], config['source'])
    vector_store = VectorStore(docs.split_docs)
    qa_chain = RetrievalChain(vector_store.vectordb)
    print("The data you provided has been loaded. Now you can chat with your data! \n Type a question and press enter to get started. To exit, type exit.")

    while True:
        question = input()
        if question.lower() == 'exit':
            print('Goodbye!!')
            break
        answer = qa_chain.get_result(question)
        print(answer["result"])


if __name__ == '__main__':
    main()
