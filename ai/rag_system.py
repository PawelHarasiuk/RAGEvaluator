from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from ai.chats import Chat


class SimpleRagSystem:
    DEFAULT_RETRIEVER_K = 4

    def __init__(self, chat: Chat, vectorstore):
        self.llm = chat.get_llm()
        k = self.DEFAULT_RETRIEVER_K
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    def _create_rag_chain(self):
        rag_prompt_template = """
        You are an AI assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.

        Question: {question}

        Context: {context}

        Answer:
        """
        RAG_PROMPT = PromptTemplate(
            template=rag_prompt_template,
            input_variables=["context", "question"]
        )

        rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": RAG_PROMPT},
            return_source_documents=True
        )

        return rag_chain

    def query_rag_system(self, query):
        rag_chain = self._create_rag_chain()
        result = rag_chain.invoke(query)

        return {
            "query": query,
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
