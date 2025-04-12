from ai.readers import DocumentReaderFactory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from typing import List

class DocumentProcessor:
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200

    def __init__(self, file_path: str):
        self.documents = self._load_document(file_path)
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def _load_document(self, file_path: str) -> any:
        reader = DocumentReaderFactory.get_reader(file_path=file_path)
        loader = reader.get_loader()
        documents = loader.load()

        return documents

    def _split_document(self) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.DEFAULT_CHUNK_SIZE,
            chunk_overlap=self.DEFAULT_CHUNK_OVERLAP,
            length_function=len
        )

        documents = self.documents
        chunks = text_splitter.split_documents(documents)

        return chunks

    def create_vectorstore(self) -> FAISS:
        chunks = self._split_document()
        vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        return vectorstore