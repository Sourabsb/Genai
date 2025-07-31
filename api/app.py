from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()


os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

app = FastAPI(
    title="Langchain API",
    description="Simple API for Langchain models",
    version="1.0",
)

add_routes(
    app,
    ChatGoogleGenerativeAI(model="gemini-2.0-flash"),
    path="/gemini"
)

model=ChatGoogleGenerativeAI(model="gemini-2.0-flash")

#ollama_model llama3
llm=Ollama(model="llama3")

prompt1=ChatPromptTemplate.from_template("Write me a short essay about {topic} with 50 words")
prompt2=ChatPromptTemplate.from_template("Write me a short poem about {topic} with 50 words")

add_routes(
    app,
    prompt1|model,
    path="/essay"
)

add_routes(
    app,
    prompt2|llm,
    path="/poem"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost",port=8000)