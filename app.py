import os
import uuid

import gradio as gr
import chromadb
import google.generativeai as genai
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = genai.GenerativeModel("gemini-2.5-flash")

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded!")


client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("rag_docs")



def extract_text(pdf):
    reader = PdfReader(pdf.name)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def chunk_text(text, chunk_size=500):
    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))

    return chunks


def upload_pdf(pdf):
    global collection

    if pdf is None:
        return "Please upload a PDF."

    text = extract_text(pdf)
    chunks = chunk_text(text)

    embeddings = embedder.encode(chunks).tolist()

    try:
        client.delete_collection("rag_docs")
    except:
        pass

    collection = client.get_or_create_collection("rag_docs")

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[str(uuid.uuid4()) for _ in chunks],
    )

    return f"Indexed {len(chunks)} chunks."


def ask(question):
    if not question:
        return "Please enter a question."

    query_embedding = embedder.encode(question).tolist()

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context = "\n".join(result["documents"][0])

    prompt = f"""
Answer ONLY using the context below.

If the answer is not found, reply:

I couldn't find the answer in the document.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.generate_content(prompt)

    return response.text



with gr.Blocks(title="RAG PDF Chatbot") as demo:

    gr.Markdown("# 📄 RAG PDF Chatbot")
    gr.Markdown("Upload a PDF and ask questions.")

    pdf = gr.File(label="Upload PDF")

    upload_button = gr.Button("Index PDF")

    status = gr.Textbox(label="Status")

    upload_button.click(
        upload_pdf,
        inputs=pdf,
        outputs=status
    )

    question = gr.Textbox(
        label="Question",
        placeholder="Ask something..."
    )

    ask_button = gr.Button("Ask")

    answer = gr.Textbox(
        label="Answer",
        lines=10
    )

    ask_button.click(
        ask,
        inputs=question,
        outputs=answer
    )

port = int(os.environ.get("PORT", 7860))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)
