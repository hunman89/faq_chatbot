# 네이버 스마트 스토어 FAQ 기반 챗봇 백엔드

> 이 프로젝트는 네이버 스마트 스토어 FAQ를 기반으로 답변하는 챗봇의 백엔드 입니다. 파이썬 기반으로 Fastapi, Milvus, openai를 이용하였습니다.

## 프로젝트 구조

- DataLoader : 초기 데이터를 가공하여 생성된 임베딩을 벡터 db에 저장한다.
- Generator : 질문과 질문의 근거가 될 수 있는 FAQ 그리고 과거 대화 내용을 이용하여 OpenAI GPT를 이용하여 답변을 생성한다.
- Embeddings : 텍스트에서 임베딩을 생성한다.
- retriever : 벡터 db에서 사용자의 질문과 유사한 FAQ 데이터를 가져온다.

## 작동 방식

```bash
git clone https://github.com/hunman89/faq_chatbot.git
cd faq_chatbot

# config.py에 apikey 삽입후.

poetry install
poetry run fastapi run

curl --request GET \
  --url http://127.0.0.1:8000/chat \
  --header 'Content-Type: application/json' \
  --data '{"query": "회원가입은 어떻게 하나요?"}'
```

## 향후 개선 사항

- 사용자별로 대화 캐싱
- 추가 질문 로직 향상
