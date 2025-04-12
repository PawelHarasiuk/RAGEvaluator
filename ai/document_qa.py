from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

from ai.readers import DocumentReader
from ai.chats import Chat


class DocumentQA:
    def __init__(self, reader: DocumentReader, chat: Chat):
        self.llm = chat.get_llm()
        loader = reader.get_loader()
        loader.load()
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        index_creator = VectorstoreIndexCreator(
            embedding=embedding_model,
            vectorstore_kwargs={"checkpoint": "all-MiniLM-L6-v2"}
        )

        self.index = index_creator.from_loaders([loader])

    def query(self, question: str):
        response = self.index.query(
            question=question,
            llm=self.llm
        )

        return response
