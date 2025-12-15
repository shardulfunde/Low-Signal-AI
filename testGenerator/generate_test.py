from langchain_cerebras import ChatCerebras
from Data_Templates.test_generation_templates import TestGenInput,Question,TestGenOutput
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
load_dotenv()

model = ChatCerebras(model="llama-3.3-70b")
parser = PydanticOutputParser(pydantic_object=TestGenOutput)


prompt = PromptTemplate(
    template="""
You are an exam paper generator.

Generate a {difficulty} level test.

Topic: {topic}
Number of questions: {num_questions}
Language: {language}

Rules:
- Only MCQ questions
- Exactly 4 options per question
- correct_index must match the correct option
- No explanations
- Output MUST follow the JSON schema below
- No extra text, no markdown

{format_instructions}
""",
    input_variables=["topic", "difficulty", "num_questions", "language"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    },
)

def generate_test_ai(payload: TestGenInput) -> TestGenOutput:
    chain = prompt | model | parser
    return chain.invoke({
        "topic": payload.topic,
        "difficulty": payload.difficulty,
        "num_questions": payload.num_questions,
        "language": payload.language,
    })