from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_cerebras import ChatCerebras
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Union
from dotenv import load_dotenv

load_dotenv()

# ==========================================
# 1. INPUT SCHEMAS (Validation)
# ==========================================
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

class QuestionItem(BaseModel):
    question: str
    options: List[str]
    correct_index: int
    # Optional: allows backend to know what the user chose
    selected_index: Optional[int] = None 

class QuizFeedbackInput(BaseModel):
    topic: str
    # We accept a list of QuestionItem objects
    questions: List[QuestionItem]
    correct_questions: List[QuestionItem]
    incorrect_questions: List[QuestionItem]


# ==========================================
# 2. OUTPUT SCHEMAS (AI Response Structure)
# ==========================================

class QuizFeedbackOutput(BaseModel):
    topic: str = Field(description="The topic of the quiz")
    understanding_level: str = Field(description="Overall understanding: Beginner, Intermediate, or Advanced")
    strengths: List[str] = Field(description="List of specific concepts the user understands well")
    weaknesses: List[str] = Field(description="List of specific concepts the user struggled with")
    suggestions: List[str] = Field(description="Actionable study tips to improve")
    feedback: str = Field(description="A short, encouraging summary paragraph")


# ==========================================
# 3. GENERATION FUNCTION
# ==========================================

def generate_quiz_feedback(payload: QuizFeedbackInput) -> dict:
    """
    Generates structured feedback based on quiz performance.
    Accepts a validated Pydantic model as input.
    """
    
    # Initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite", 
        temperature=0.3
    )
    
    parser = PydanticOutputParser(pydantic_object=QuizFeedbackOutput)
    
    # Helper to format list data cleanly for the prompt
    def format_questions(q_list: List[QuestionItem]):
        if not q_list:
            return "None"
        
        formatted_text = ""
        for item in q_list:
            formatted_text += f"- Question: {item.question}\n"
            # If we have the user's wrong answer index, it helps the AI explain WHY they were wrong
            if item.selected_index is not None and item.selected_index != item.correct_index:
                wrong_answer = item.options[item.selected_index]
                formatted_text += f"  (User incorrectly chose: {wrong_answer})\n"
                
        return formatted_text

    # Format inputs for the prompt
    questions_str = format_questions(payload.questions)
    correct_str = format_questions(payload.correct_questions)
    incorrect_str = format_questions(payload.incorrect_questions)

    prompt = PromptTemplate(
        template="""
You are an expert learning mentor. 
Analyze the user's quiz attempt on the topic: "{topic}" and generate structured feedback.

DATA:
---
ALL QUESTIONS:
{questions}

CORRECTLY ANSWERED:
{correct_questions}

INCORRECTLY ANSWERED:
{incorrect_questions}
---

INSTRUCTIONS:
1. Analyze the "INCORRECTLY ANSWERED" section to identify specific gaps in knowledge.
2. Analyze the "CORRECTLY ANSWERED" section to identify concepts they have mastered.
3. Provide specific "Suggestions" on what to study next based on the errors.
4. Output MUST be valid JSON matching the schema below.

{format_instructions}
""",
        input_variables=["topic", "questions", "correct_questions", "incorrect_questions"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "topic": payload.topic,
            "questions": questions_str,
            "correct_questions": correct_str,
            "incorrect_questions": incorrect_str
        })
        
        return result.model_dump()
        
    except Exception as e:
        print(f"Error generating feedback: {e}")
        return {
            "topic": payload.topic,
            "understanding_level": "Error",
            "strengths": [],
            "weaknesses": [],
            "suggestions": ["Please try again."],
            "feedback": "Could not generate feedback at this time."
        }