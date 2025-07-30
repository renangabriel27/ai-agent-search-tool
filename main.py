from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from google import genai
import os

load_dotenv()

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
messages = [HumanMessage(content="Gostaria de saber mais sobre transurfing?")]
response = llm.invoke(messages)
print(response.content)
