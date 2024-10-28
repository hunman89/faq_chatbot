
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pymilvus import MilvusClient
from data_loader import DataLoader
from generator import Generator
from retriever import Retriever


class Chat(BaseModel):
    query: str


app = FastAPI()
db_client = MilvusClient("data/milvus.db")
DataLoader(db_client)
retriever = Retriever(db_client)
generator = Generator()


@app.get("/chat")
async def chat(chat: Chat):
    context = retriever.retrieve(chat.query)
    return StreamingResponse(generator.generate(context, chat.query), media_type="text/event-stream")
