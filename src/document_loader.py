from langchain.document_loaders import PyPDFLoader, YoutubeLoader
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentLoader:
    doc_type = None
    doc_loader = None
    source = None
    docs = None
    split_docs = []
    
    def __init__(self, doc_type: str, source: str) -> None:
        """
        Initialise the document loader based on the type of the source document

        Args:
            doc_type (str): Type of the document - PDFs and URLs are currently supported
            source (str): The path to the source document
        """
        self.doc_type = doc_type
        if self.doc_type == 'pdf':
            self.doc_loader = PyPDFLoader(source)
        elif self.doc_type == 'youtube':
            self.doc_loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v="+source)
        elif self.doc_type == 'url':
            self.doc_loader = WebBaseLoader(source)
        self.load_docs()
        self.split_documents()
    
    def load_docs(self) -> None:
        """
        Load the documents from the source
        """
        self.docs = self.doc_loader.load()
    
    def split_documents(self) -> None:
        """
        Split the documents into chunks using a text splitter
        """
        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=450,
            chunk_overlap=0, 
            separators=["\n\n", "\n", " ", ""]
        )
        self.split_docs = r_splitter.split_documents(self.docs)
