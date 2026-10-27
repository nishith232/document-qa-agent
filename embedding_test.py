
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

print("Total chunks:", len(chunks))
print("Embedding shape:", embeddings.shape)
print("First embedding:", embeddings[0][:10])
