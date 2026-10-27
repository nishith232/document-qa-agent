
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

reader = PdfReader("sample.pdf")

text = ""

for page in reader.pages:
    text += page.extract_text() or ""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)
