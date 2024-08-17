import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings # 변경된 import 문
import pickle
import torch

# pdf_texts 로드
with open('pdf_texts_paragraphs.pkl', 'rb') as f:
    pdf_texts = pickle.load(f)

# 임베딩 로드
embedding_output_dir = "C:\\Users\\lge\\3학년 2학기 도전학기\\embeddings"
embeddings = torch.load(os.path.join(embedding_output_dir, "embeddings.pt"))

# KR-SBERT 모델 로드 (임베딩 함수로 사용)
model_name = 'snunlp/KR-SBERT-V40K-klueNLI-augSTS'
embedding_function = SentenceTransformerEmbeddings(model_name=model_name) # SentenceTransformerEmbeddings 사용

# ChromaDB 설정
persist_directory = "C:\\Users\\lge\\3학년 2학기 도전학기\\chroma_db"
chroma_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

# 데이터 추가
for i, (file_name, text) in enumerate(pdf_texts):
    # embeddings는 이미 생성되어 있으므로, embed_documents를 호출하지 않고 직접 전달합니다.
    chroma_db.add_texts([text], metadatas=[{"source": file_name}], ids=[str(i)], embeddings=[embeddings[i]])

# 저장된 문서 수 확인
num_documents = len(chroma_db._collection.get()["documents"])
print(f"Number of documents in ChromaDB: {num_documents}")
