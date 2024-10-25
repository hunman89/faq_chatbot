from model import RAGModel


def main():
    rag_model = RAGModel()
    while True:
        query = input("질문을 입력하세요 (종료: 'exit'): ")
        if query.lower() == 'exit':
            break
        answer = rag_model.get_answer(query)
        print(f"답변: {answer}")


if __name__ == "__main__":
    main()
