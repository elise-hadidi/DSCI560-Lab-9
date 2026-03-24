import os
import io
import re
import html
import argparse
from typing import List, Dict
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

from PyPDF2 import PdfReader
try:
    import fitz
    from PIL import Image
    import pytesseract
    FITZ_AVAILABLE = True
except Exception:
    FITZ_AVAILABLE = False

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings

data_path       = "../Data/"
chunk_param     = 500
overlap         = 100
delimiter       = "\n"
string_min      = 200

load_dotenv()

OPEN_AI_KEY     = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL       = "gpt-3.5-turbo"

client = OpenAI()

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = html.unescape(s)
    s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_pages_pypdf2(pdf_bytes: bytes) -> List[str]:
    pages = []
    try:
        r = PdfReader(io.BytesIO(pdf_bytes))
        for p in r.pages:
            pages.append(p.extract_text() or "")
    except Exception:
        return []
    return pages


def extract_pages_fitz_ocr(pdf_bytes: bytes, ocr_threshold: int = string_min) -> List[str]:
    if not FITZ_AVAILABLE:
        return []
    pages = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            t = page.get_text("text") or ""
            if len(t.strip()) >= ocr_threshold:
                pages.append(t)
            else:
                try:
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = pytesseract.image_to_string(img) or ""
                    pages.append(ocr_text if ocr_text.strip() else t)
                except Exception:
                    pages.append(t)
        doc.close()
    except Exception:
        return []
    return pages


def pdf_file_to_page_texts(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        pdf_bytes = f.read()
    pages = extract_pages_pypdf2(pdf_bytes)
    if (not pages or all(not p.strip() for p in pages)) and FITZ_AVAILABLE:
        pages = extract_pages_fitz_ocr(pdf_bytes)
    if not pages:
        return [{"source_pdf": os.path.basename(path), "page_number": 1, "page_text": ""}]
    return [
        {"source_pdf": os.path.basename(path), "page_number": i + 1, "page_text": clean_text(p or "")}
        for i, p in enumerate(pages)
    ]


def load_all_pdfs(folder_path: str) -> List[Dict]:
    all_pages = []
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {folder_path}")
    for pdf_file in pdf_files:
        print(f"  Loading: {pdf_file}")
        all_pages.extend(pdf_file_to_page_texts(os.path.join(folder_path, pdf_file)))
    return all_pages


def pages_to_chunks(page_rows: List[Dict],
                    chunk_size: int = chunk_param,
                    chunk_overlap: int = overlap,
                    separator: str = delimiter) -> List[str]:
    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = []
    for pr in page_rows:
        text = pr.get("page_text", "") or ""
        if not text.strip():
            continue
        for c in splitter.split_text(text):
            c = c.strip()
            if len(c) >= 10:
                chunks.append(c)
    return chunks


def get_vectorstore(text_chunks: List[str]) -> FAISS:
    print(f"  Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_chain(vectorstore: FAISS):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    def answer_question(question: str) -> str:
        docs = retriever.invoke(question)
        context = "\n\n".join(doc.page_content.strip() for doc in docs if doc.page_content.strip())
        if not context:
            return "I could not find relevant content in the document."

        response = client.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=512,
            messages=[
                {"role": "system", "content": "Answer questions using only the provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        )
        return response.choices[0].message.content.strip()

    return answer_question


def clean_answer(raw: str) -> str:
    return raw.strip()


def run_chat(chain) -> None:
    print("\n=== Chatbot Ready ===")
    print("Ask a question about your document, or type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        print(f"\nBot: {chain(user_input)}\n")


def main(pdf_path: str) -> None:
    if os.path.isdir(pdf_path):
        print(f"Loading all PDFs from: {pdf_path}")
        pages = load_all_pdfs(pdf_path)
    else:
        print(f"Loading PDF: {pdf_path}")
        pages = pdf_file_to_page_texts(pdf_path)
    print(f"  Extracted {len(pages)} pages.")
    print("Chunking text...")
    chunks = pages_to_chunks(pages)
    print(f"  Created {len(chunks)} chunks.")
    print("Building vector store...")
    vectorstore = get_vectorstore(chunks)
    print("  Vector store ready.")
    print("Initializing chain...")
    chain = get_chain(vectorstore)
    run_chat(chain)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", type=str, default=data_path)
    args = ap.parse_args()
    main(args.pdf)
