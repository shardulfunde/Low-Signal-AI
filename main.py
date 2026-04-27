from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io

# ---- Your imports ----
from Chatbot.chatbot import Ai_stream
from testGenerator.generate_test import generate_test_ai
from Data_Templates.test_generation_templates import TestGenInput, TestGenOutput
from Data_Templates.learning_path_templates import (
    LearningPathInput, LearningPathOutPut,
    TopicList, Topic, TopicDetail, TTSRequest
)
from learningpath import (
    create_learning_path, create_topic_list,
    create_topic_detail, topic_detail_event_stream
)
from sarvam_api import generate_sarvam_tts
from test_analysis import analyze_test_service, TestAnalysisInput, TestAnalysisOutput
from learning_path_feedback import generate_quiz_feedback, QuizFeedbackInput, QuizFeedbackOutput

# ---- App init ----
app = FastAPI()

# ---- CORS (FIXED) ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://lowsignal.onrender.com",  # your frontend
        "http://localhost:3000",           # local dev
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---- Preflight handler (IMPORTANT) ----
@app.options("/{full_path:path}")
async def preflight_handler():
    return Response(status_code=200)

# ---- Health ----
@app.get("/")
def health():
    return {"status": "ok"}

# ---- Chat Streaming ----
@app.get("/chat/stream")
def chat_stream(question: str):
    def event_generator():
        try:
            for token in Ai_stream(question):
                yield f"data: {token}\n\n"
        except Exception as e:
            yield f"data: Error: {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

# ---- Test Generation ----
@app.post("/test/generate", response_model=TestGenOutput)
def generate_test(payload: TestGenInput):
    try:
        result = generate_test_ai(payload)
        return result
    except Exception as e:
        print(f"Error in test generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ---- Learning Path ----
@app.post("/learning_path/generate", response_model=LearningPathOutPut)
def generate_learning_path(payload: LearningPathInput):
    try:
        return create_learning_path(payload)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learning_path/generate/topic_list", response_model=TopicList)
def generate_topic_list(payload: LearningPathInput):
    try:
        return create_topic_list(payload)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learning_path/generate/topic_detail", response_model=Topic)
def generate_topic_detail(payload: TopicDetail):
    try:
        return create_topic_detail(payload)
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learning_path/generate/topic_detail/stream")
def stream_topic_detail(payload: TopicDetail):
    return StreamingResponse(
        topic_detail_event_stream(payload),
        media_type="text/event-stream"
    )

# ---- TTS ----
@app.post("/generate_tts/")
def generate_tts(request: TTSRequest):
    try:
        audio_data = generate_sarvam_tts(request.text, request.language)

        if not audio_data:
            return Response(content="Failed to generate audio", status_code=500)

        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type="audio/wav"
        )

    except Exception as e:
        print(f"Server Error: {str(e)}")
        return Response(content=str(e), status_code=500)

# ---- Quiz Feedback ----
@app.post("/generate_feedback", response_model=QuizFeedbackOutput)
async def generate_feedback_route(payload: QuizFeedbackInput):
    try:
        result = generate_quiz_feedback(payload)

        if isinstance(result, dict) and result.get("understanding_level") == "Error":
            return result

        return result

    except Exception as e:
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ---- Test Analysis ----
@app.post("/test/analyze", response_model=TestAnalysisOutput)
def analyze_test(payload: TestAnalysisInput):
    try:
        return analyze_test_service(payload)
    except Exception as e:
        print(f"Error analyzing test: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
