from langchain_community.document_loaders import CSVLoader, TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

# loader = CSVLoader(file_path='realistic_restaurant_reviews.csv')
loader = TextLoader(file_path='test.txt')
loader.load()
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

index_creator = VectorstoreIndexCreator(
    embedding=embedding_model,
    vectorstore_kwargs={"checkpoint": "all-MiniLM-L6-v2"}
)

index = index_creator.from_loaders([loader])

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="AIzaSyCqLLHtLpz-1w1P7bg6m9es-JeenjJ1g44")

query = input('Give input: ')
retriever = index.query(
    question=query,
    llm=llm
)
print(retriever)


