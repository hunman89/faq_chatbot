import openai

from config import GPT_MODEL, OPENAI_API_KEY


class Generator:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY
        self.conversation_history = []

    async def generate(self, context, query):
        context_data = {'conversation_history': self.conversation_history,
                   'context': context}
        messages = self.build_messages(context_data, query)
        stream = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=messages,
            stream=True
        )
        answer = ""

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                answer += chunk.choices[0].delta.content

        self.update_history(query, answer)
        
    @staticmethod
    def build_messages(context_data, query):
        return [
            {"role": "system", "content": "You are an assistant that answers questions based on the provided context. If the context does not have sufficient information to answer the question, Main Answer is '저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.'"},
            {"role": "user", "content": f"Context: {context_data}\nQuestion: {query}"},
            {"role": "assistant", "content": "After providing an answer, also provide an additional question labeled 'Suggested Question' that the user might be curious about based on the provided context. Format as follows:\n\nMain Answer: [Your answer here]\nSuggested Question: [Your suggested question here]"}
        ]

    def update_history(self, query, answer):
        self.conversation_history.append({'question': query, 'answer': answer})