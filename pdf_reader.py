
from pypdf import PdfReader

reader = PdfReader("sample.pdf")

print("Total pages:", len(reader.pages))

for i, page in enumerate(reader.pages, start=1):
    text = page.extract_text() or ""
    print(f"\n--- Page {i} ---")
    print(text)
