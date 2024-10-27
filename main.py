from fastapi import FastAPI, Response
from pydantic import BaseModel
from model import RAGModel


class Chat(BaseModel):
    query: str


app = FastAPI()


@app.post("/chat")
async def chat(chat: Chat):
    model = RAGModel()
    return Response(model.get_answer(chat.query), media_type="text/plain")
