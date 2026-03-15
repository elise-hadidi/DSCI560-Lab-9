from dotenv import load_dotenv
import os

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

import os
import io
import re
import html
import argparse
from typing import List, Dict


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


data_path = "/Data/Ads Cookbook.pdf"  

chunk_param = 500
overlap = 100
delimiter = "\n"
string_min = 200   


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

    rows = []
    for i, p in enumerate(pages):
        rows.append({
            "source_pdf": os.path.basename(path),
            "page_number": i + 1,
            "page_text": clean_text(p or "")
        })
    return rows
def pages_to_chunks(page_rows: List[Dict],
                    chunk_size: int = chunk_param,
                    overlap: int = overlap,
                    separator: str = delimiter,
                    text_only: bool = True
                    ) -> Union[List[str], List[Dict]]:

    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        overlap=overlap,
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

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def main(pdf_path: str):
    
    pages = pdf_file_to_page_texts(pdf_path)
    
    chunks = pages_to_chunks(pages, chunk_size=chunk_param, overlap=overlap)
    
    vs = get_vectorstore(chunks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract PDF, chunk text (500 chars), create FAISS vectorstore (OpenAI embeddings).")
    ap.add_argument("--pdf", type=str, default=data_path, help="Path to PDF file")
    args = ap.parse_args()
    main(args.pdf)