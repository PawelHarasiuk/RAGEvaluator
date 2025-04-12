from document import DocumentProcessor
from rag_system import SimpleRagSystem
from chats import GeminiChat

document_processor = DocumentProcessor('../data/test.txt')
vectorstore = document_processor.create_vectorstore()

rag_system = SimpleRagSystem(GeminiChat(), vectorstore)

prompt = 'give me latest news!!!'
print(rag_system.query_rag_system(prompt))
