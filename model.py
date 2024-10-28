from retriever import Retriever
from generator import Generator


class RAGModel:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()
        self.conversation_history = []

    def get_answer(self, query):
        context = self.retriever.retrieve(query)
        answer = self.generator.generate(
            {'conversation_history': self.conversation_history, 'context': context}, query)
        self.conversation_history.append(
            {'question': query, 'answer': answer})
        return answer
