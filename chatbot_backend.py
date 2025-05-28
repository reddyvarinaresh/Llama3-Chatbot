from langchain_groq.chat_models import ChatGroq
from langchain_core.pydantic_v1 import BaseModel, Field
from utils import math_exec, general_exec
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
os.environ["GROQ_API_KEY"] = os.getenv("GROQQ_API_KEY")

class Question_Type(BaseModel):
    question_type: str = Field(description="The type of question. One of two: math or general")

question_router_llm = ChatGroq(model="llama-3.1-8b-instant")
question_router = question_router_llm.with_structured_output(Question_Type)
general_model = ChatGroq(model="llama-3.1-8b-instant")


def chatbot(question_type, chat_history, question):
    if question_type == "math":
        print("Using math model...")
        return math_exec(general_model, question, chat_history)
    else:
        print("Using general model...")
        return general_exec(general_model, question, chat_history)

def chat(chat_history):
    question = chat_history[-1]["content"]
    question_router_formatted = """
    Identify the type of question whether it is math or general. If there is some kind of math involved then it is math or else it is general.
    Question: {question}""".format(question=question)
    question_type = question_router.invoke(question_router_formatted).question_type
    chat_history = chatbot(question_type, chat_history, question)
    return chat_history[-1]["content"]

