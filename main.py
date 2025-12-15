from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from Chatbot.chatbot import Ai_stream
from testGenerator.generate_test import generate_test_ai
from Data_Templates.test_generation_templates import TestGenInput, TestGenOutput
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

class Query(BaseModel):
    question : str
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {"Status":"Ok"}

@app.get("/chat/stream")
def chat_stream(question:str):
    def event_generator():
        for token in Ai_stream(question):
            yield f"data: {token}\n\n"
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
    
@app.post("/test/generate",response_model=TestGenOutput)
def generate_test(payload:TestGenInput):
    result = generate_test_ai(payload)
    return result
