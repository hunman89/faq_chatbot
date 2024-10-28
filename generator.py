import openai

from config import GPT_MODEL, OPENAI_API_KEY


class Generator:
    def __init__(self):
        openai.api_key = OPENAI_API_KEY

    def generate(self, context, query):
        messages = [
            {"role": "system", "content": "You are an assistant that answers questions based on the provided context."},
            {"role": "user", "content": f"Context: {context}\nQuestion: {query}"},
            {"role": "assistant",
                "content": "After providing an answer labeled 'Main Answer', also provide an additional question labeled 'Suggested Question' that the user might find interesting based on the provided context. Format as follows:\n\nMain Answer: [Your answer here]\nSuggested Question: [Your suggested question here]"}
        ]
        response = openai.chat.completions.create(
            model=GPT_MODEL,
            messages=messages
        )
        return response.choices[0].message.content.strip()
