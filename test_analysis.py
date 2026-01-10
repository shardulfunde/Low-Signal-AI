import os
from dotenv import load_dotenv
from typing import List
from langchain_cerebras import ChatCerebras
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# ==========================================
# 1. DATA MODELS
# ==========================================

class QuestionResult(BaseModel):
    """
    Represents a single question's result from the test.
    Contains the question text, the answer provided by the user, and the correct answer.
    """
    question: str
    selected_option_index: int  # Index of the answer given by user (-1 if skipped)
    correct_option_index: int   # Index of the correct option
    options: List[str]          # List of all options to map indices to text

class TestAnalysisInput(BaseModel):
    topic: str
    language: str
    results: List[QuestionResult]

class TestAnalysisOutput(BaseModel):
    score_commentary: str = Field(description="A brief, encouraging comment on the score.")
    weak_concepts: List[str] = Field(description="List of specific sub-topics the user struggled with.")
    strengths: List[str] = Field(description="List of concepts the user understood well.")
    study_plan: List[str] = Field(description="3 actionable bullet points to improve.")


# ==========================================
# 2. MODEL SETUP
# ==========================================
# Using Gemini Flash for fast and accurate analysis
analysis_model = ChatCerebras(model="qwen-3-235b-a22b-instruct-2507")


# ==========================================
# 3. SERVICE FUNCTION
# ==========================================

def analyze_test_service(payload: TestAnalysisInput) -> TestAnalysisOutput:
    parser = PydanticOutputParser(pydantic_object=TestAnalysisOutput)

    # 1. Format the Input Data into a Clear Text Summary for the AI
    # This transforms indices back into readable text: "Question -> User Answer -> Correct Answer"
    test_summary = ""
    
    for idx, item in enumerate(payload.results):
        # Determine User Answer Text
        if 0 <= item.selected_option_index < len(item.options):
            user_ans_text = item.options[item.selected_option_index]
        else:
            user_ans_text = "No Answer / Skipped"
            
        # Determine Correct Answer Text
        correct_ans_text = item.options[item.correct_option_index]
        
        # Determine Status
        is_correct = "CORRECT" if item.selected_option_index == item.correct_option_index else "INCORRECT"
        
        test_summary += f"""
        Question {idx + 1}: {item.question}
        - User Answer: {user_ans_text}
        - Correct Answer: {correct_ans_text}
        - Result: {is_correct}
        ----------------------------------
        """

    # 2. Define the Prompt
    prompt = PromptTemplate(
        template="""
        You are an expert personalized tutor. 
        Analyze the following test results to help the student improve.

        Context:
        - Topic: {topic}
        - Language: {language}

        Test Results:
        {test_data_summary}

        Task:
        1. Compare the User's Answer vs the Correct Answer for each question.
        2. Identify specific patterns in their mistakes (e.g., confused concept A with B).
        3. Highlight the concepts they answered correctly.
        4. Create a specific, actionable 3-step study plan.

        Output Rules:
        - Be encouraging but direct about mistakes.
        - Output MUST be valid JSON matching the schema below.
        - No markdown formatting (like ```json), just the raw JSON string.

        {format_instructions}
        """,
        input_variables=["topic", "language", "test_data_summary"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )

    # 3. Create and Invoke Chain
    chain = prompt | analysis_model | parser
    
    return chain.invoke({
        "topic": payload.topic,
        "language": payload.language,
        "test_data_summary": test_summary
    })