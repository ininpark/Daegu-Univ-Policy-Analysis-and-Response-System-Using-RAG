import os
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# OpenAI API 설정
openai_api_key = "실제 키로 대체"  
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.2, model_name="gpt-4o")

# persist_directory는 save_to_chroma_db.py에서 사용한 디렉토리와 동일해야 합니다.
persist_directory = "C:\\Users\\lge\\3학년 2학기 도전학기\\chroma_db"

# KR-SBERT 모델 로드 
model_name = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'
embedding_function = SentenceTransformerEmbeddings(model_name=model_name)

# ChromaDB 설정 (벡터 저장소 로드)
chroma_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

# 프롬프트 템플릿 정의
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            **당신은 사용자에게 도움이 되는 대구대학교 학칙 및 규정 기반 응답 어시스턴트입니다.** 
            다음의 문맥을 사용하여 질문에 답변하십시오.
            * **문서의 내용을 기반으로 정확하고 상세하게 답변**하십시오. 
            * **관련 법령 조항을 언급**하여 답변의 신뢰성을 높이십시오.
            * **문서에서 제공된 정보만을 사용하여 응답**을 생성하십시오.
            * **만약 답을 모른다면 "정보를 찾을 수 없습니다"라고 답변**하고, 답변을 지어내거나 추측하지 마십시오.
            \n\n
            {context}
            """
        ),
        ("human", "{question}"),
    ]
)

# RAG 응답 생성 함수
def generate_rag_response(query, vector_store, llm, prompt_template, top_k=5, max_context_length=500):
    # 검색 실행
    results = vector_store.similarity_search(query, k=top_k)
   
    # 문맥 생성
    context = "\n\n".join([res.page_content for res in results])
   
    # 문맥 길이 조정
    if len(context) > max_context_length:
        context = context[:max_context_length]
   
    # 체인 생성
    chain = LLMChain(
        llm=llm, 
        prompt=prompt_template
    )

    # 체인 실행 및 응답 생성
    result = chain.run({
        "context": context,
        "question": query
    })
    return result

# 검색어 정의
query = "입학합격을 통보 받은 사람이 중간에 입학포기를 선언하면 등록금을 돌려받을 수 있나요?"

# RAG 응답 생성 및 출력
response = generate_rag_response(query, chroma_db, llm, prompt_template, top_k=5, max_context_length=500)
print("RAG Response:")
print(response)
