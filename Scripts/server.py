import os
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from app_opensource import (
    pdf_file_to_page_texts,
    pages_to_chunks,
    get_vectorstore,
    get_chain,
    clean_answer,
    chunk_param,
    overlap
)

app = Flask(__name__, static_folder="../Web")
CORS(app, resources={r"/*": {"origins": "*"}})

UPLOAD_FOLDER = "../Data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

active_chain = None


@app.route("/")
def index():
    return send_from_directory("../Web", "index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global active_chain

    if "pdfs" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400

    files = request.files.getlist("pdfs")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files selected"}), 400

    all_pages = []
    saved_files = []

    for f in files:
        if f and f.filename.lower().endswith(".pdf"):
            filename = secure_filename(f.filename)
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            save_path = os.path.join(UPLOAD_FOLDER, unique_name)
            f.save(save_path)
            saved_files.append(filename)
            all_pages.extend(pdf_file_to_page_texts(save_path))

    if not all_pages:
        return jsonify({"error": "Could not extract text from uploaded files"}), 400

    chunks = pages_to_chunks(all_pages, chunk_size=chunk_param)
    vectorstore = get_vectorstore(chunks)
    active_chain = get_chain(vectorstore)

    return jsonify({
        "message": f"Processed {len(saved_files)} file(s) into {len(chunks)} chunks.",
        "files": saved_files
    })


@app.route("/chat", methods=["POST"])
def chat():
    global active_chain

    if active_chain is None:
        return jsonify({"error": "No documents loaded. Please upload a PDF first."}), 400

    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' field"}), 400

    question = data["question"].strip()
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    answer = active_chain(question)

    return jsonify({"answer": answer})


if __name__ == "__main__":
    print("Starting server at http://127.0.0.1:5000")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=True, port=5000)