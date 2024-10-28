
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from generator import Generator
from retriever import Retriever


class Chat(BaseModel):
    query: str


app = FastAPI()
retriever = Retriever()
generator = Generator()


@app.get("/chat")
async def chat(chat: Chat):
    context = retriever.retrieve(chat.query)
    return StreamingResponse(generator.generate(context, chat.query), media_type="text/event-stream")
