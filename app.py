import os
from flask import Flask, request, render_template_string, make_response, Response, render_template
import fitz
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import requests
import json
import csv
import pandas as pd
import io
from datetime import datetime
import uuid
 
# ----- CONFIGURATION -----
GROQ_API_KEY = "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"
# Fixed: Use a valid Groq model name
GROQ_MODEL = "llama-3.3-70b-versatile"  # Alternative: "llama-3.1-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"
CHUNK_SIZE = 4000
 
# ----- EMBEDDING MODEL -----
embedder = SentenceTransformer('all-MiniLM-L6-v2')
 
# ----- GLOBAL STATE -----
CORPUS = []            # List of (chunk, metadata)
CORPUS_FILES = []      # List of uploaded filenames
EMBEDDINGS = None      # np.array of all embeddings (same order as CORPUS)
INDEX = None           # FAISS index for semantic search
LATEST_EXTRACTION_OBJ = {}  # Dict of latest extracted fields, for download
EXTRACTION_HISTORY = []     # List to store all extraction records with metadata
 
# ----- UTILITIES -----
def generate_prompt(text: str) -> str:
    chunk = text[:CHUNK_SIZE]
    return f"""
You are a medical document extractor. Extract the following fields in STRICT JSON format:
 
- patient_name
- patient_id
- age (number)
- Gender (Male/Female/Other)
- dob (YYYY-MM-DD format if available)
- Adress(full adress)
- phone_number
- Marital Status (Single/Married/Divorced/Widowed)
- Blood_group (A+/A-/B+/B-/O+/O-/AB+/AB-)
- allergies
- past_medical_history(list)
- Vaccination_history(list)
- Dose(number)
- Frequency

- physical_Activity (Sedentary/Lightly/Moderately/veryactive)
- Smoking_status (Never/Former/Current)
- Alcohol_consumption  (None/Occasional/Moderate/Heavy)
- hospital_id
- record_date (YYYY-MM-DD)
- Blood_Pressure (Systolic/Diastolic)
- Heart_Rate (BPM)
- Temperature (Celsius)
- Respiratory_Rate (Breaths per minute)
- diagnosis_notes
- review_of_systems (dict of system: [symptoms])
- current_medications (list)
- LAB_RESULTS(DICT OF LAB_NAME:VALUE)
- imaging_results(list of dicts with keys:type,date,findings)
- Doses
- Frequency
- Reason for _doseas
- Side_effects
- Prescribing_doctor(Name)
 
Rules:
- Return only VALID JSON (no markdown or explanation).
- If fields are not found, use "" for strings or [] for lists accordingly.
- Dates must be in YYYY-MM-DD format if available.
- Ensure JSON is syntactically valid and strictly follows the requested schema.
 
Text:
\"\"\"{chunk}\"\"\"
"""
 
def extract_text_from_pdf(file_storage) -> str:
    file_storage.seek(0)
    pdf = fitz.open(stream=file_storage.read(), filetype="pdf")
    text = "\n".join(page.get_text() for page in pdf)
    return text
 
def extract_medical_fields_with_groq(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a medical extraction assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "response_format": {"type": "json_object"},
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        if response.status_code == 401:
            raise Exception("Invalid API key or unauthorized access. Please check your GROQ_API_KEY.")
        elif response.status_code == 400:
            raise Exception("Invalid request. Please check the model name and request format.")
        else:
            raise Exception(f"HTTP Error {response.status_code}: {e}")
    except Exception as e:
        raise Exception(f"Error calling Groq API: {e}")
 
def extract_text_chunks_from_text(text, chunk_size=400):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def save_extraction_record(extracted_data, filename):
    """Save extraction record to history with metadata"""
    global EXTRACTION_HISTORY
    
    record = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "extracted_data": extracted_data,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    EXTRACTION_HISTORY.append(record)
    return record["id"]
 
def add_to_corpus(text, source_filename):
    global CORPUS, CORPUS_FILES, EMBEDDINGS, INDEX
    chunks = extract_text_chunks_from_text(text, chunk_size=400)
    CORPUS.extend([(chunk, {"file": source_filename}) for chunk in chunks])
    if source_filename not in CORPUS_FILES:
        CORPUS_FILES.append(source_filename)
    all_chunks = [c[0] for c in CORPUS] if CORPUS else []
    if all_chunks:
        EMBEDDINGS = embedder.encode(all_chunks, show_progress_bar=False)
        INDEX = faiss.IndexFlatL2(EMBEDDINGS.shape[1])
        INDEX.add(np.array(EMBEDDINGS, dtype=np.float32))
    else:
        EMBEDDINGS = None
        INDEX = None
 
def semantic_search(query, k=3):
    if EMBEDDINGS is None or INDEX is None or not CORPUS:
        return []
    query_emb = embedder.encode([query])
    distances, indices = INDEX.search(np.array(query_emb, dtype=np.float32), k)
    return [CORPUS[i][0] for i in indices[0] if i < len(CORPUS)]
 
def ask_llm_with_rag(question, context):
    prompt = f"""
Answer the medical question below using ONLY the provided context. Be honest if answer is not found.
 
Context:
\"\"\"{context}\"\"\"
 
Question: {question}
Answer:
"""
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a clinical Q&A assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error getting answer: {e}"

def _error_response(msg):
    """Return JSON for API requests, HTML for browser requests"""
    return (
        Response(json.dumps({'error': msg}), mimetype='application/json')
        if request.accept_mimetypes['application/json'] >= request.accept_mimetypes['text/html']
        else render_template_string('<h2>Error</h2><p>{{ message }}</p>', message=msg)
    )
 
# ----- FLASK APP -----
app = Flask(__name__)
from flask_cors import CORS
CORS(app) 

# Enable CORS for all routes
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,Accept')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/', defaults={'path': ''}, methods=['OPTIONS'])
@app.route('/<path:path>', methods=['OPTIONS'])
def handle_options(path):
    """Handle preflight requests"""
    return '', 200

# Main route to serve the HTML interface
@app.route("/")
def index():
    """Main interface route - serves the HTML frontend or API info"""
    # Check if it's an API request (JSON preferred) or browser request
    if request.accept_mimetypes['application/json'] >= request.accept_mimetypes['text/html']:
        # API response
        info = {
            "message": "Healthcare AI Extraction API",
            "version": "1.0",
            "model_used": GROQ_MODEL,
            "endpoints": {
                "/": "Main web interface and API info",
                "/admin": "Admin interface for PDF upload (GET/POST) - supports both browser and JSON",
                "/user": "User interface for Q&A (GET/POST) - supports both browser and JSON",
                "/clear_vectors": "Clear vector store (POST)",
                "/extract_records": "Get all extraction records (GET)",
                "/download_json": "Download latest extraction as JSON",
                "/download_csv": "Download latest extraction as CSV",
                "/download_excel": "Download latest extraction as Excel",
                "/download_all_json": "Download all records as JSON",
                "/download_all_csv": "Download all records as CSV",
                "/download_all_excel": "Download all records as Excel"
            },
            "usage": {
                "web_interface": "Visit / for the main web interface",
                "postman_extract": "POST /admin with form-data containing 'pdf_file' and Accept: application/json header",
                "postman_ask": "POST /user with form-data 'question' or JSON body {'question': 'your question'} and Accept: application/json header"
            }
        }
        return Response(json.dumps(info, indent=2), mimetype='application/json')
    else:
        # Serve the HTML template
        return render_template('index.html')

# Original admin HTML template for backward compatibility
ADMIN_HTML = '''
<!doctype html>
<title>Healthcare AI Extraction (/admin)</title>
<h2>Upload a Healthcare PDF for Extraction (Admin)</h2>
<form method=post enctype=multipart/form-data>
    <input type=file name=pdf_file accept=".pdf" required>
    <input type=submit value=Extract>
</form>
<form method="post" action="{{ url_for('clear_vectors') }}"
      onsubmit="return confirm('Are you sure you want to clear the vector store? This will remove all indexed documents from memory.');">
    <button type="submit" style="background-color: #b71c1c; color: white; margin-top:10px;">Clear Vector Store</button>
</form>
{% if result %}
    <h3>Extracted Fields (Groq):</h3>
    <pre>{{ result_json }}</pre>
    <form method="get" action="{{ url_for('download_json') }}">
      <button type="submit">Download JSON</button>
    </form>
    <form method="get" action="{{ url_for('download_csv') }}">
      <button type="submit">Download CSV</button>
    </form>
    <form method="get" action="{{ url_for('download_excel') }}"> <button type="submit">Download Excel</button> </form>
{% endif %}
{% if error %}
    <p style="color:red;">{{ error }}</p>
{% endif %}
<hr>
<!-- Notice: No user page link here as per your request -->
'''
 
USER_HTML = '''
<!doctype html>
<title>Healthcare AI Q&A (/user)</title>
<h2>Ask a Medical Question (User)</h2>
<form method=post>
    <input name=question size=100 required>
    <input type=submit value="Get Answer">
</form>
{% if answer %}
    <strong>Answer (Groq+RAG):</strong>
    <pre>{{ answer }}</pre>
{% endif %}
{% if context %}
    <details>
    <summary>Context Chunks (Top {{context_chunks}}):</summary>
    <pre>{{ context }}</pre>
    </details>
{% endif %}
<hr>
{% if docs %}
    <p>Indexed documents: <ul>{% for doc in docs %}<li>{{doc}}</li>{% endfor %}</ul></p>
{% else %}
    <p style="color:red;">No documents indexed yet. Please upload in <code>/admin</code> first.</p>
{% endif %}
'''

@app.route("/admin", methods=["GET", "POST"])
def admin():
    global LATEST_EXTRACTION_OBJ
    
    if request.method == 'GET':
        # Show the last extracted (even after page reload) for GET requests
        to_display = LATEST_EXTRACTION_OBJ if LATEST_EXTRACTION_OBJ else None
        result_json = json.dumps(to_display, indent=2, ensure_ascii=False) if to_display else None
        
        return (
            Response(json.dumps({
                "latest_extraction": to_display,
                "extraction_history_count": len(EXTRACTION_HISTORY),
                "message": "Use POST to upload PDF file with 'pdf_file' parameter",
                "current_model": GROQ_MODEL
            }, indent=2, ensure_ascii=False), mimetype='application/json')
            if request.accept_mimetypes['application/json'] >= request.accept_mimetypes['text/html']
            else render_template_string(ADMIN_HTML, result=to_display, result_json=result_json, error=None)
        )
    
    # POST request logic
    result, error = None, None
    
    file = request.files.get("pdf_file")
    if not file:
        return _error_response("No file uploaded.")
    
    try:
        file.seek(0)
        text = extract_text_from_pdf(file)
        prompt = generate_prompt(text)
        extraction = extract_medical_fields_with_groq(prompt)
        add_to_corpus(text, file.filename)
        
        try:
            result = json.loads(extraction)
        except Exception:
            result = extraction
        
        # Save for download and history
        LATEST_EXTRACTION_OBJ = result if isinstance(result, dict) else {}
        record_id = save_extraction_record(LATEST_EXTRACTION_OBJ, file.filename)
        
        # Prepare response
        output = {
            "success": True,
            "extracted_data": result,
            "filename": file.filename,
            "record_id": record_id,
            "message": "PDF processed successfully",
            "model_used": GROQ_MODEL
        }
        
        json_output = json.dumps(output, indent=2, ensure_ascii=False)
        result_json = json.dumps(result, indent=2, ensure_ascii=False)
        
        return (
            Response(json_output, mimetype='application/json')
            if request.accept_mimetypes['application/json'] >= request.accept_mimetypes['text/html']
            else render_template_string(ADMIN_HTML, result=result, result_json=result_json, error=error)
        )
        
    except Exception as e:
        return _error_response(f"Error: {e}")

@app.route('/extract_records', methods=["GET"])
def extract_records():
    """Get all extraction records"""
    return (
        Response(json.dumps({
            'success': True,
            'total_records': len(EXTRACTION_HISTORY),
            'records': EXTRACTION_HISTORY
        }, indent=2, ensure_ascii=False), mimetype='application/json')
        if request.accept_mimetypes['application/json'] >= request.accept_mimetypes['text/html']
        else Response(json.dumps(EXTRACTION_HISTORY, indent=2, ensure_ascii=False), mimetype='application/json')
    )
 
@app.route('/clear_vectors', methods=["POST"])
def clear_vectors():
    global CORPUS, CORPUS_FILES, EMBEDDINGS, INDEX, LATEST_EXTRACTION_OBJ, EXTRACTION_HISTORY
    CORPUS = []
    CORPUS_FILES = []
    EMBEDDINGS = None
    INDEX = None
    LATEST_EXTRACTION_OBJ = {}
    EXTRACTION_HISTORY = []
    
    return (
        Response(json.dumps({'success': True, 'message': 'Vector store and extraction history cleared successfully'}), mimetype='application/json')
        if request.accept_mimetypes['application/json'] >= request.accept_mimetypes['text/html']
        else ("", 204)  # No content for browser, current page will reload
    )
 
@app.route('/download_json')
def download_json():
    if not LATEST_EXTRACTION_OBJ:
        return _error_response("No extraction available.")
    
    response = make_response(json.dumps(LATEST_EXTRACTION_OBJ, indent=2))
    response.headers['Content-Type'] = 'application/json'
    response.headers['Content-Disposition'] = 'attachment; filename=extracted_data.json'
    return response
 
@app.route('/download_csv')
def download_csv():
    if not LATEST_EXTRACTION_OBJ:
        return _error_response("No extraction available.")
    
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=LATEST_EXTRACTION_OBJ.keys())
    writer.writeheader()
    writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in LATEST_EXTRACTION_OBJ.items()})
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=extracted_data.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response

@app.route('/download_excel')
def download_excel():
    if not LATEST_EXTRACTION_OBJ:
        return _error_response("No extraction available.")

    # Convert dict/list values to JSON strings so they fit in single Excel cells
    row = {
        k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
        for k, v in LATEST_EXTRACTION_OBJ.items()
    }

    # Create a single-row DataFrame
    df = pd.DataFrame([row])

    # Write to in-memory Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='extracted_data')
    output.seek(0)

    response = make_response(output.read())
    response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    response.headers['Content-Disposition'] = 'attachment; filename=extracted_data.xlsx'
    return response

# New routes for downloading all records
@app.route('/download_all_json')
def download_all_json():
    if not EXTRACTION_HISTORY:
        return _error_response("No extraction records available.")
    
    response = make_response(json.dumps(EXTRACTION_HISTORY, indent=2, ensure_ascii=False))
    response.headers['Content-Type'] = 'application/json'
    response.headers['Content-Disposition'] = 'attachment; filename=all_extraction_records.json'
    return response

@app.route('/download_all_csv')
def download_all_csv():
    if not EXTRACTION_HISTORY:
        return _error_response("No extraction records available.")
    
    output = io.StringIO()
    if EXTRACTION_HISTORY:
        # Flatten the structure for CSV
        flattened_records = []
        for record in EXTRACTION_HISTORY:
            flat_record = {
                'record_id': record.get('id', ''),
                'timestamp': record.get('timestamp', ''),
                'filename': record.get('filename', ''),
                'created_at': record.get('created_at', '')
            }
            # Add all extracted data fields
            extracted_data = record.get('extracted_data', {})
            for key, value in extracted_data.items():
                flat_record[key] = json.dumps(value) if isinstance(value, (dict, list)) else value
            flattened_records.append(flat_record)
        
        if flattened_records:
            fieldnames = flattened_records[0].keys()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened_records)
    
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=all_extraction_records.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response

@app.route('/download_all_excel')
def download_all_excel():
    if not EXTRACTION_HISTORY:
        return _error_response("No extraction records available.")
    
    # Flatten the structure for Excel
    flattened_records = []
    for record in EXTRACTION_HISTORY:
        flat_record = {
            'record_id': record.get('id', ''),
            'timestamp': record.get('timestamp', ''),
            'filename': record.get('filename', ''),
            'created_at': record.get('created_at', '')
        }
        # Add all extracted data fields
        extracted_data = record.get('extracted_data', {})
        for key, value in extracted_data.items():
            flat_record[key] = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
        flattened_records.append(flat_record)
    
    # Create DataFrame
    df = pd.DataFrame(flattened_records)
    
    # Write to in-memory Excel
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='all_records')
    output.seek(0)
    
    response = make_response(output.read())
    response.headers['Content-Type'] = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    response.headers['Content-Disposition'] = 'attachment; filename=all_extraction_records.xlsx'
    return response

@app.route("/user", methods=["GET", "POST"])
def user_qa():
    if request.method == 'GET':
        return (
            Response(json.dumps({
                "indexed_documents": CORPUS_FILES,
                "total_chunks": len(CORPUS),
                "extraction_records_count": len(EXTRACTION_HISTORY),
                "message": "Use POST with 'question' parameter to ask questions",
                "model_used": GROQ_MODEL
            }, indent=2), mimetype='application/json')
            if request.accept_mimetypes['application/json'] >= request.accept_mimetypes['text/html']
            else render_template_string(USER_HTML, answer=None, context=None, context_chunks=0, docs=CORPUS_FILES)
        )
    
    # POST request logic
    answer = context = None
    context_chunks = 0
    
    question = request.form.get("question") or (request.get_json() or {}).get("question")
    
    if not question:
        return _error_response("Question parameter is required.")
    
    if question and CORPUS:
        top_chunks = semantic_search(question, k=3)
        context = "\n---\n".join(top_chunks)
        context_chunks = len(top_chunks)
        answer = ask_llm_with_rag(question, context)
    elif not CORPUS:
        answer = "No documents available for Q&A. Please upload in admin first."
    
    # Prepare response
    output = {
        "success": True,
        "question": question,
        "answer": answer,
        "model_used": GROQ_MODEL
    }
    
    json_output = json.dumps(output, indent=2, ensure_ascii=False)
    
    return (
        Response(json_output, mimetype='application/json')
        if request.accept_mimetypes['application/json'] >= request.accept_mimetypes['text/html']
        else render_template_string(
            USER_HTML,
            answer=answer,
            context=context,
            context_chunks=context_chunks,
            docs=CORPUS_FILES
        )
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=7007)