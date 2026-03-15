# DSCI560-Lab-9

**Team Members:** Elise Hadidi (1137648541), Jordan Davies (1857892197), Ryan Silva (6463166471) 
**Team Number:**  17

## Overview
A Q&A chatbot that extracts text from PDF documents, splits it into chunks, generates embeddings using open-source HuggingFace models, stores them in a FAISS vector database, and answers user questions through a web interface. 

## Folder Structure
```
DSCI560-Lab-9/
│
├── Scripts/
│   ├── app_p1.py
│   ├── app_opensource.py
│   └── server.py  
│
├── Web/
│   └── index.html     
│
├── Data/
│   └── Ads cookbook .pdf
│
├── requirements.txt
├── README.md        
└── .env  
```

## Files
* app.py — main pipeline with OpenAI
* app_opensource.py - main pipeline with open source resources
* server.py — Flask backend
* index.html — web interface with PDF upload and chat window

## Requirements
### Install dependencies with:
```
pip install -r requirements.txt
```

## How to Run
1. Start the Flask backend (from DSCI560-Lab-9/Scripts):
```
python server.py
```

2. Open your browser and go to:
```
http://127.0.0.1:5000
```

3. Upload one or multiple PDF(s) using the upload box, click "Process", then type a question.
