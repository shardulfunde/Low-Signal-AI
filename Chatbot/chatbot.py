from langchain_cerebras import ChatCerebras
from langchain_google_genai import ChatGoogleGenerativeAI   
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


model = ChatCerebras(model="qwen-3-235b-a22b-instruct-2507",streaming=True)


def Ai_stream(question:str):
    for chunk in model.stream(question):
        if(chunk.content):
            yield chunk.content