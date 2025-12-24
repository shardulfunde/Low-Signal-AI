from langchain_cerebras import ChatCerebras
from dotenv import load_dotenv

load_dotenv()

model = ChatCerebras(model="qwen-3-235b-a22b-instruct-2507",streaming=True)


def Ai_stream(question:str):
    for chunk in model.stream(question):
        if(chunk.content):
            yield chunk.content