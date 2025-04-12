from abc import ABC, abstractmethod
from langchain_google_genai import ChatGoogleGenerativeAI


class Chat(ABC):
    @abstractmethod
    def get_llm(self):
        pass


class GeminiChat(Chat):
    def get_llm(self):
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                      google_api_key="")
