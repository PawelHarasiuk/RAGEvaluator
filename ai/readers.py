from abc import ABC, abstractmethod
from langchain_community.document_loaders import CSVLoader, TextLoader, PyPDFLoader


class DocumentReader(ABC):
    @abstractmethod
    def get_loader(self):
        pass


class CSVReader(DocumentReader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_loader(self):
        loader = CSVLoader(file_path=self.file_path)
        return loader


class TextReader(DocumentReader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_loader(self):
        loader = TextLoader(file_path=self.file_path)
        return loader


class PDFReader(DocumentReader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_loader(self):
        loader = PyPDFLoader(file_path=self.file_path)
        return loader


class DocumentReaderFactory:
    @staticmethod
    def get_reader(file_path: str) -> DocumentReader:
        if file_path.endswith('.csv'):
            return CSVReader(file_path)
        elif file_path.endswith('.txt'):
            return TextReader(file_path)
        elif file_path.endswith('.pdf'):
            return PDFReader(file_path)
        else:
            raise ValueError(f"Unsupported file extension for file {file_path}")
