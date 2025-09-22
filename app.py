from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

app = FastAPI()

# Load LLM
model = OllamaLLM(model="llama3.2")

# Prompt Template
template = """
You are an expert in answering questions about a pizza restaurant.

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


@app.get("/")
async def get_home():
    with open("index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        question = await websocket.receive_text()
        if question.lower() == "q":
            await websocket.close()
            break
        reviews = retriever.invoke(question)
        result = chain.invoke({"reviews": reviews, "question": question})
        await websocket.send_text(result)

