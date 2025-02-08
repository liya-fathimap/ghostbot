from flask import Flask, request, jsonify, render_template
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import cohere  # Import Cohere
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load Sentence Transformer model
hf_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Set Cohere API Key (Replace with your actual key)
COHERE_API_KEY = "V15GCCKaznDA2wWEy07wH8QMdGmExr98TbxAZ7A4"
cohere_client = cohere.Client(COHERE_API_KEY)  # Initialize Cohere client

# Store embeddings globally
chunks, chunk_embeddings = [], None

# PDF file to process
PDF_PATH = "Tinkerhack.pdf"

# Function to extract text from a PDF
def extract_text_from_pdf(file_path):
    try:
        pdf_reader = PdfReader(file_path)
        text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        return text
    except Exception as e:
        print("Error reading PDF:", str(e))
        return ""

# Function to split text and generate embeddings
def process_text_and_generate_embeddings(text, chunk_size=200):
    global chunks, chunk_embeddings
    if not text.strip():
        print("Error: No text extracted from PDF!")
        return
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    chunk_embeddings = hf_model.encode(chunks)

# Function to find the most relevant text chunk
def get_similar_chunk(user_input):
    if  chunk_embeddings is not list():
        return "PDF is not loaded or text processing failed."

    question_embedding = hf_model.encode([user_input])
    similarities = cosine_similarity(question_embedding, chunk_embeddings)[0]
    top_match_idx = np.argmax(similarities)
    return chunks[top_match_idx]

# Function to generate a response using Cohere API
def generate_response(prompt):
    try:
        print(prompt)
        response = cohere_client.generate(
            model="command",  # You can also try "command-r" for better responses
            prompt=prompt,
            max_tokens=50,
            temperature=0.7
        )
        return response.generations[0].text.strip()  # Extract response text
    except cohere.CohereError as e:
        print("Cohere API error:", str(e))
        return "I'm having trouble generating a response right now."

# Preload PDF data at startup
if os.path.exists(PDF_PATH):
    print("Processing PDF:", PDF_PATH)
    pdf_text = extract_text_from_pdf(PDF_PATH)
    if pdf_text:
        process_text_and_generate_embeddings(pdf_text)
        print("PDF processing complete.")
    else:
        print("No text extracted from PDF.")
else:
    print(f"Error: {PDF_PATH} not found!")

# Home route
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", uploaded=True)

# Chat route (Process user messages)
@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.form.get("question", "").strip()
    
    if not user_question:
        return jsonify({"response": "Please enter a valid message."})

    print(f"User Question: {user_question}")  # Debugging output

    similar_text = get_similar_chunk(user_question)
    print(f"Similar Text Found: {similar_text}")  # Debugging output

    bot_response = generate_response(f"Reply to this message in my texting style: {similar_text}")
    print(f"Bot Response: {bot_response}")  # Debugging output

    return jsonify({"response": bot_response})

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
