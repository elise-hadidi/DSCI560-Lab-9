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
│   ├── openai_app.py
│   ├── pdf_extractor_updated.py
│   ├── app_opensource.py
│   └── server.py  
│
├── Web/
│   └── index.html     
│
├── Data/
│   ├── uploads/
│   │   └──0f4f3aa0986740d7a4cde3217c05c8ad_Ads_cookbook_.pdf
│   └── Ads cookbook .pdf
│
├── requirements.txt
├── README.md        
└── .env  
```

## Files
* openai_app.py — main pipeline with OpenAI
* pdf_extractor_updated.py - PDF extractor
* app_opensource.py - main pipeline with open source resources
* server.py — Flask backend
* index.html — web interface with PDF upload and chat window

## Requirements
### Install dependencies with:
```
pip install -r requirements.txt
```

## How to Run
1. Set up .env file with OpenAI API key

3. Start the Flask backend (from DSCI560-Lab-9/Scripts):
```
python server.py
```

3. Open your browser and go to:
```
http://127.0.0.1:5000
```

4. Upload one or multiple PDF(s) using the upload box, click "Process", then type a question.
