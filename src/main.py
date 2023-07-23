import os
import openai
import sys
import argparse
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
from document_loader import DocumentLoader

_ = load_dotenv(find_dotenv()) # read local .env file
parser = argparse.ArgumentParser(description="Command Line Arguments for the Retrival Augmented Generation System",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--type", help="type of the source", default="pdf", choices=["pdf", "youtube", "url"], type=str)
parser.add_argument("--source", help="source of the information", type=str)

openai.api_key  = os.environ['OPENAI_API_KEY']

def main():
    args = parser.parse_args()
    config = vars(args)
    doc_loader = DocumentLoader(config["type"], config['source'])

if __name__ == '__main__':
    main()