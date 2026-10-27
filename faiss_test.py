
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

reader = PdfReader("sample.pdf")

chunks = []

for page in reader.pages:
    text = page.extract_text() or ""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks.extend(splitter.split_text(text))

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)
embeddings = np.array(embeddings, dtype="float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

question = "What programming languages do I know?"
question_embedding = model.encode([question])
question_embedding = np.array(question_embedding, dtype="float32")

distances, indices = index.search(question_embedding, k=2)

print("Question:", question)

for i in indices[0]:
    print("\nRelevant chunk:")
    print(chunks[i])
