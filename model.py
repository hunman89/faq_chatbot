from retriever import Retriever
from generator import Generator


class RAGModel:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()
        self.conversation_history = []

    def get_answer(self, query):
        context = self.retriever.retrieve(query)
        if context:
            answer = self.generator.generate(
                {'conversation_history': self.conversation_history, 'context': context}, query)
        else:
            answer = "저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다."
        self.conversation_history.append(
            {'question': query, 'answer': answer})
        return answer
