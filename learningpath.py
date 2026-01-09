
import json
from langchain_cerebras import ChatCerebras
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser,JsonOutputParser
from Data_Templates.learning_path_templates import LearningPathInput,Topic,LearningPathOutPut,TopicList,TopicDetail

load_dotenv()


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")


topic_parser = PydanticOutputParser(pydantic_object=Topic)
topic_list_parser = PydanticOutputParser(pydantic_object=TopicList)

topic_planner_prompt = PromptTemplate(
    template="""
You are an expert curriculum designer.

Task:
Generate a learning topic plan for the learner described below.

Input:
Subject: {subject}
Learner age: {year_old}
Preferred language: {preferred_language}
Focus areas: {focus_areas}

Strict rules:
- Generate **exactly 6 to 8 topics**.
- Topics must be ordered from **basic to advanced**.
- Topic names must be **short, clear, and non-overlapping**.
- Do NOT include explanations, questions, examples, or numbering.
- If focus areas are provided, ensure **each focus area is covered by at least one topic**.
- Use **only** the preferred language:
  - en → English
  - hi → Hindi
  - mr → Marathi
- Do NOT add extra fields or text.

Output rules:
- You must generate a JSON object that is an INSTANCE of the schema.
- Do NOT return the schema or its definitions.
- Do NOT include "properties", "required", or any field descriptions.
- Populate the "topics" field with actual topic names.
- Return ONLY the final JSON object.


{format_instructions}
""",
    input_variables=["subject", "year_old", "preferred_language", "focus_areas"],
    partial_variables={
        "format_instructions": topic_list_parser.get_format_instructions()
    },
)

from langchain_core.prompts import PromptTemplate
model = ChatCerebras(model="qwen-3-235b-a22b-instruct-2507")
model2 = ChatCerebras(model="qwen-3-235b-a22b-instruct-2507",streaming=True)
topic_expander_prompt = PromptTemplate(
    template="""
You are an expert tutor.

Task:
Create very detailed explanation and practice questions for the given topic, tailored to the learner's profile.

Global learner context:
Subject: {subject}
Learner age: {year_old}
Preferred language: {preferred_language}

Topic to expand:
Topic name: {topic_name}

Strict rules:
- Explain the topic clearly and simply, appropriate for the learner's age,and in detail
- Explanation must be focused only on this topic.
- Create **exactly 5 easy practice questions** suitable for beginners.
- Do NOT include answers unless the schema requires them.
- Do NOT reference other topics.
- Do NOT add extra fields or commentary.
- Use **only** the preferred language:
  - en → English
  - hi → Hindi
  - mr → Marathi
Output rules:
- You must generate a JSON object that is an INSTANCE of the schema.
- Do NOT return the schema or its definitions.
- Do NOT include "$defs", "properties", or "required".
- Populate all required fields with actual values.
- Return only valid JSON.


{format_instructions}
""",
    input_variables=[
        "subject",
        "year_old",
        "preferred_language",
        "topic_name",
    ],
    partial_variables={
        "format_instructions": topic_parser.get_format_instructions()
    },
)


topic_planner_chain = topic_planner_prompt | model | topic_list_parser
topic_expander_chain = topic_expander_prompt | model2 
topic_expander_parser_chain = topic_expander_prompt | model | topic_parser

def create_learning_path(payload:LearningPathInput)->LearningPathOutPut:
    result = topic_planner_chain.invoke(
        payload.model_dump()
    )

    topic_expander = topic_expander_parser_chain.batch([
        {
            "subject": payload.subject,
            "year_old": payload.year_old,
            "preferred_language": payload.preferred_language,
            "topic_name": topic,
        }
        for topic in result.topics
    ])

    topics_detailed: list[Topic] = topic_expander
    learning_path = LearningPathOutPut(
        topics=topics_detailed,
        additional_resources=None,
    )
    return learning_path

def create_topic_list(payload:LearningPathInput) -> TopicList:
    topic_list = topic_planner_chain.invoke(payload.model_dump())
    return topic_list

def create_topic_detail(payload:TopicDetail) -> Topic:
    topic_detail = topic_expander_parser_chain.invoke(
        {
            "subject": payload.payload.subject,
            "year_old": payload.payload.year_old,
            "preferred_language": payload.payload.preferred_language,
            "topic_name": payload.topic_name,
        }
    )
    return topic_detail
def topic_detail_event_stream(payload: TopicDetail):
    parser = JsonOutputParser(pydantic_object=Topic)
    streaming_chain = topic_expander_prompt | model2 | parser

    input_data = {
        "subject": payload.payload.subject,
        "year_old": payload.payload.year_old,
        "preferred_language": payload.payload.preferred_language,
        "topic_name": payload.topic_name,
    }

    last_sent_length = 0
    final_data = {} 

    try:
        for chunk in streaming_chain.stream(input_data):
            final_data = chunk 
            
            if "explanation" in chunk and chunk["explanation"]:
                current_text = chunk["explanation"]
                
                if len(current_text) > last_sent_length:
                    new_content = current_text[last_sent_length:]
                    last_sent_length = len(current_text)
                    
                    yield f"data: {json.dumps({
                        'type': 'explanation_chunk', 
                        'data': new_content
                    })}\n\n"

        if "practice_questions" in final_data:
             for q in final_data["practice_questions"]:
                yield f"data: {json.dumps({
                    'type': 'question', 
                    'data': q
                })}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"