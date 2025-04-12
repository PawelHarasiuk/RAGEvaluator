from abc import ABC, abstractmethod
from langchain_community.document_loaders import CSVLoader, TextLoader, PyPDFLoader


class DocumentReader(ABC):
    @abstractmethod
    def get_loader(self):
        pass


class CSVReader(DocumentReader):
    def get_loader(self):
        loader = CSVLoader(file_path='realistic_restaurant_reviews.csv')
        return loader


class TextReader(DocumentReader):
    def get_loader(self):
        loader = TextLoader(file_path='test.txt')
        return loader


class PDFReader(DocumentReader):
    def get_loader(self):
        loader = PyPDFLoader(file_path='')
        return loader

class DocumentReaderFactory:
    @staticmethod
    def get_reader(file_path: str) -> DocumentReader:
        if file_path.endswith('.csv'):
            return CSVReader()
        elif file_path.endswith('.txt'):
            return TextReader()
        elif file_path.endswith('.pdf'):
            return PDFReader()
        else:
            raise ValueError(f"Unsupported file extension for file {file_path}")
