from retriever import Retriever
from generator import Generator


class RAGModel:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()

    def get_answer(self, query):
        retrieved_docs = self.retriever.retrieve(query)
        context = " ".join(retrieved_docs)
        answer = self.generator.generate(context, query)
        return answer
