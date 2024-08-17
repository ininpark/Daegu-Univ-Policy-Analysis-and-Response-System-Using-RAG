from sentence_transformers import SentenceTransformer
import pickle
import torch
import os

# KR-SBERT 모델 로드
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

# pdf_texts 로드
with open('pdf_texts_paragraphs.pkl', 'rb') as f:
    pdf_texts = pickle.load(f)

# 청킹된 텍스트만 추출
chunked_texts = [text for _, text in pdf_texts]

# 임베딩 생성
embeddings = model.encode(chunked_texts)

# 임베딩 확인
print(f"Generated {len(embeddings)} embeddings.")
print(embeddings[0])  # 첫 번째 임베딩 출력

# 임베딩 저장 디렉토리 설정
embedding_output_dir = "C:\\Users\\lge\\3학년 2학기 도전학기\\embeddings"
os.makedirs(embedding_output_dir, exist_ok=True)

# 임베딩 저장
torch.save(embeddings, os.path.join(embedding_output_dir, "embeddings.pt"))

print(f"Embeddings saved to {embedding_output_dir}.")
