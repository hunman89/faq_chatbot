import openai

from config import GPT_MODEL, OPENAI_API_KEY


class Generator:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY

    def generate(self, context, query):
        messages = [
            {"role": "system", "content": "You are an assistant that answers questions based on the provided context. If the context does not have sufficient information to answer the question, Main Answer is '저는 스마트 스토어 FAQ를 위한 챗봇입니다. 스마트 스토어에 대한 질문을 부탁드립니다.'"},
            {"role": "user", "content": f"Context: {context}\nQuestion: {query}"},
            {"role": "assistant",
                "content": "After providing an answer, also provide an additional question labeled 'Suggested Question' that the user might be curious about based on the provided context. Format as follows:\n\nMain Answer: [Your answer here]\nSuggested Question: [Your suggested question here]"}
        ]
        response = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=messages
        )
        return response.choices[0].message.content.strip()
