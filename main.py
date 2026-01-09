from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from Chatbot.chatbot import Ai_stream
from testGenerator.generate_test import generate_test_ai
from Data_Templates.test_generation_templates import TestGenInput, TestGenOutput
from Data_Templates.learning_path_templates import LearningPathInput, LearningPathOutPut,TopicList,Topic,TopicDetail
from learningpath import create_learning_path, create_topic_list, create_topic_detail, topic_detail_event_stream
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

@app.post("/learning_path/generate",response_model=LearningPathOutPut)
def generate_learning_path(payload:LearningPathInput):
    return create_learning_path(payload)
    
@app.post("/learning_path/generate/topic_list",response_model=TopicList)
def generate_topic_list(payload:LearningPathInput):
    return create_topic_list(payload)

@app.post("/learning_path/generate/topic_detail",response_model=Topic)
def generate_topic_detail(payload:TopicDetail):
    return create_topic_detail(payload)

@app.post("/learning_path/generate/topic_detail/stream")
def stream_topic_detail(payload:TopicDetail):
    return StreamingResponse(
        topic_detail_event_stream(payload),
        media_type="text/event-stream"
    )