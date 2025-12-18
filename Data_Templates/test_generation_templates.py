from pydantic import BaseModel,Field
from typing import List,Literal

class TestGenInput(BaseModel):
    topic:str
    difficulty:Literal["easy","medium","hard"]
    num_questions: int = Field(default=5,gt=0,le=20)
    language:Literal["en","hi","mr"]
    
class Question(BaseModel):
    question:str
    options:List[str]=Field(description="Give 4 options")
    correct_index:int
    
class TestGenOutput(BaseModel):
    topic:str
    difficuly:str
    questions:List[Question]