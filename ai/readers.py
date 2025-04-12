from abc import ABC, abstractmethod
from langchain_community.document_loaders import CSVLoader, TextLoader


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
