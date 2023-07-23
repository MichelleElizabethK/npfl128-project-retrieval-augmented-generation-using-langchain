from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain.document_loaders import WebBaseLoader


class DocumentLoader:
    doc_type = None
    doc_loader = None
    source = None
    docs = None
    
    def __init__(self, doc_type: str, source: str) -> None:
        self.doc_type = doc_type
        if self.doc_type == 'pdf':
            self.doc_loader = PyPDFLoader(source)
        elif self.doc_type == 'youtube':
            save_dir="docs/youtube/"
            self.doc_loader = GenericLoader(
                YoutubeAudioLoader([source],save_dir),
                OpenAIWhisperParser()
            )
        elif self.doc_type == 'url':
            self.doc_loader = WebBaseLoader(source)
        self.docs = self.doc_loader.load()
