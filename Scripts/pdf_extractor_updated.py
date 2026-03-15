from dotenv import load_dotenv
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

import io
import re
import html
import argparse
from typing import List, Dict, Union  # added Union (was missing)

from PyPDF2 import PdfReader
try:
    import fitz
    from PIL import Image
    import pytesseract
    FITZ_AVAILABLE = True
except Exception:
    FITZ_AVAILABLE = False

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# ── NEW imports for Section 2d/2e ────────────────────────────────────────────
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


# ── Configuration ─────────────────────────────────────────────────────────────

data_path   = "/Data/Ads Cookbook.pdf"
chunk_param = 500
overlap     = 100
delimiter   = "\n"
string_min  = 200


# ── Text Cleaning ─────────────────────────────────────────────────────────────

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = html.unescape(s)
    s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# ── PDF Extraction ────────────────────────────────────────────────────────────

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

    rows = []
    for i, p in enumerate(pages):
        rows.append({
            "source_pdf": os.path.basename(path),
            "page_number": i + 1,
            "page_text": clean_text(p or "")
        })
    return rows


# ── Chunking ──────────────────────────────────────────────────────────────────

def pages_to_chunks(page_rows: List[Dict],
                    chunk_size: int = chunk_param,
                    overlap: int = overlap,
                    separator: str = delimiter,
                    text_only: bool = True) -> Union[List[str], List[Dict]]:

    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=overlap,   # NOTE: correct param name is chunk_overlap
        length_function=len
    )
    chunks_texts = []
    chunks_with_meta = []

    for pr in page_rows:
        text = pr.get("page_text", "") or ""
        if not text.strip():
            continue
        split = splitter.split_text(text)

        for cid, c in enumerate(split):
            c = c.strip()
            if not c:
                continue
            if text_only:
                if len(c) < 10:
                    continue
                chunks_texts.append(c)
            else:
                chunks_with_meta.append({
                    "source_pdf": pr.get("source_pdf"),
                    "page_number": pr.get("page_number"),
                    "chunk_id": cid,
                    "text": c
                })

    return chunks_texts if text_only else chunks_with_meta

def pages_to_chunk_texts(page_rows: List[Dict],
                         chunk_size: int = chunk_param,
                         overlap: int = overlap,
                         separator: str = delimiter) -> List[str]:
    return pages_to_chunks(page_rows,
                           chunk_size=chunk_size,
                           overlap=overlap,
                           separator=separator,
                           text_only=True)


# ── Vector Store (Section 2c) ─────────────────────────────────────────────────

def get_vectorstore(text_chunks: List[str]) -> FAISS:
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# ── Conversation Chain (Section 2d) ──────────────────────────────────────────

def get_conversation_chain(vectorstore: FAISS) -> ConversationalRetrievalChain:
    """
    Creates a conversational retrieval chain that:
      - Uses GPT-3.5-turbo as the LLM
      - Pulls the top 3 most relevant chunks from the vector store per query
      - Remembers conversation history with ConversationBufferMemory
        so follow-up questions are understood in context
    """
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0          # 0 = factual/deterministic answers
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",   # must match what the chain expects
        return_messages=True         # stores messages as chat objects, not plain text
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )

    return chain


# ── Interactive Q&A Loop (Section 2e) ─────────────────────────────────────────

def run_chat(chain: ConversationalRetrievalChain) -> None:
    """
    Prompts the user for questions in a loop, passes each question through
    the conversation chain, and prints the answer. Type 'exit' to quit.
    """
    print("\n=== Chatbot Ready ===")
    print("Ask a question about the document, or type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        response = chain({"question": user_input})
        print(f"\nBot: {response['answer']}\n")


# ── Driver (Section 2e) ───────────────────────────────────────────────────────

def main(pdf_path: str) -> None:
    print(f"Loading PDF: {pdf_path}")
    pages = pdf_file_to_page_texts(pdf_path)
    print(f"  Extracted {len(pages)} pages.")

    print("Chunking text...")
    chunks = pages_to_chunks(pages, chunk_size=chunk_param, overlap=overlap)
    print(f"  Created {len(chunks)} chunks.")

    print("Building vector store (this may take a moment)...")
    vectorstore = get_vectorstore(chunks)
    print("  Vector store ready.")

    print("Initializing conversation chain...")
    chain = get_conversation_chain(vectorstore)

    run_chat(chain)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="PDF Q&A chatbot using OpenAI embeddings + GPT-3.5-turbo."
    )
    ap.add_argument("--pdf", type=str, default=data_path, help="Path to PDF file")
    args = ap.parse_args()
    main(args.pdf)