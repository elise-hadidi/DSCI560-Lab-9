import os
import io
import re
import html
import warnings
warnings.filterwarnings("ignore")

from PyPDF2 import PdfReader
import duckdb
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

try:
    import fitz
    from PIL import Image
    import pytesseract
    FITZ_AVAILABLE = True
except:
    FITZ_AVAILABLE = False

DB_PATH = "../Data/chatbot.duckdb"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "deepset/roberta-base-squad2"
chunk_param = 500
overlap = 100
delimiter = "\n"
string_min = 200


def init_db():
    con = duckdb.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS pdf_pages (
        id INTEGER PRIMARY KEY, source VARCHAR,
        page_num INTEGER, page_text VARCHAR)""")
    con.execute("""CREATE TABLE IF NOT EXISTS pdf_chunks (
        id INTEGER PRIMARY KEY, source VARCHAR,
        chunk_id INTEGER, chunk_text VARCHAR)""")
    con.close()


def save_pages_to_db(page_rows):
    con = duckdb.connect(DB_PATH)
    con.execute("DELETE FROM pdf_pages")
    for i, row in enumerate(page_rows):
        con.execute("INSERT INTO pdf_pages VALUES (?, ?, ?, ?)",
            [i, row["source_pdf"], row["page_number"], row["page_text"]])
    con.close()


def save_chunks_to_db(chunks, source):
    con = duckdb.connect(DB_PATH)
    con.execute("DELETE FROM pdf_chunks")
    for i, chunk in enumerate(chunks):
        con.execute("INSERT INTO pdf_chunks VALUES (?, ?, ?, ?)",
            [i, source, i, chunk])
    con.close()


def clean_text(s):
    if not isinstance(s, str):
        return ""
    s = html.unescape(s)
    s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E]", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_pages_pypdf2(pdf_bytes):
    pages = []
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for p in reader.pages:
            pages.append(p.extract_text() or "")
    except:
        return []
    return pages


def extract_pages_fitz_ocr(pdf_bytes):
    if not FITZ_AVAILABLE:
        return []
    pages = []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            t = page.get_text("text") or ""
            if len(t.strip()) >= string_min:
                pages.append(t)
            else:
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_out = pytesseract.image_to_string(img) or ""
                    pages.append(ocr_out if ocr_out.strip() else t)
                except:
                    pages.append(t)
        doc.close()
    except:
        return []
    return pages


def get_pdf_text(path):
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


def load_all_pdfs(folder_path):
    all_pages = []
    pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {folder_path}")
    for fname in pdf_files:
        print(f"  Loading: {fname}")
        all_pages.extend(get_pdf_text(os.path.join(folder_path, fname)))
    return all_pages


def get_text_chunks(page_rows, chunk_size=chunk_param, chunk_overlap=overlap, separator=delimiter):
    splitter = CharacterTextSplitter(
        separator=separator, chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, length_function=len
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


def get_vectorstore(text_chunks):
    print(f"  Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_conversation_chain(vectorstore):
    print(f"  Loading LLM: {LLM_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForQuestionAnswering.from_pretrained(LLM_MODEL)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    def answer_question(question):
        docs = retriever.invoke(question)
        best_answer = ""
        best_score = float("-inf")

        for doc in docs:
            context = doc.page_content.strip()
            if not context:
                continue

            inputs = tokenizer(question, context, return_tensors="pt",
                               truncation=True, max_length=512, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)

            start_idx = torch.argmax(outputs.start_logits)
            end_idx = torch.argmax(outputs.end_logits) + 1
            score = (outputs.start_logits[0][start_idx] + outputs.end_logits[0][end_idx - 1]).item()

            if end_idx > start_idx and score > best_score:
                tokens = inputs["input_ids"][0][start_idx:end_idx]
                candidate = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                if candidate:
                    best_score = score
                    best_answer = candidate

        return best_answer if best_answer else "I could not find a clear answer in the document."

    return answer_question