from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os, threading

from pypdf import PdfReader
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import logging
logging.getLogger("uvicorn.access").disabled = True


def chunk_text(text, chunk_size=2000, max_chunks=5):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
        if len(chunks) >= max_chunks:
            break
    return chunks




app = FastAPI()

# ---------------------------
# Setup
# ---------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

progress = {"percent": 0, "msg": "Idle"}
result_data = {}

def set_progress(p, msg):
    progress["percent"] = p
    progress["msg"] = msg
    print(f"[PROGRESS] {p}% - {msg}")

# ---------------------------
# LLM
# ---------------------------
llm = OllamaLLM(
    model="gemma3:4b",
    temperature=0.1,
    repeat_penalty=1.2
)

prompt = PromptTemplate(
    input_variables=["context", "questions"],
    template="""
You are an exam-answer generator.

STRICT RULES:
1. Use ONLY PDF context.
2. Answer question by question.
3. Do NOT list all questions first.
4. After each question, immediately give its answer.
5. Follow format EXACTLY.
6. Short answers only (2–4 lines).

FORMAT (MUST FOLLOW):

Q1. <question>
Ans: <answer>

Q2. <question>
Ans: <answer>

Q3. <question>
Ans: <answer>

PDF Context:
{context}

Questions:
{questions}
"""
)


# ---------------------------
# Background worker
# ---------------------------
def process_pdf(path):
    try:
        set_progress(10, "Reading PDF...")
        reader = PdfReader(path)
        full_text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        chunks = chunk_text(full_text)
        print(f"[CHUNKS] Created: {len(chunks)} chunks")
        text = "\n".join(chunks)
        print("[SERVER] Page loaded")
        print(f"[PDF] Total pages: {len(reader.pages)}")
        print(f"[CHUNKS] Text length: {len(text)} characters")
        
        set_progress(30, "Extracting questions...")
        questions = [l for l in text.split("\n") if l.strip().endswith("?")]
        if not questions:
            questions = ["Explain the topic"]
        questions_text = "\n".join(questions[:5])

        set_progress(60, "AI thinking...")
        result = llm.invoke(
            prompt.format(context=text[:12000], questions=questions_text)
        )

        set_progress(90, "Formatting answer...")
        answers = result.strip().split("\n\n")

        result_data["answers"] = answers
        set_progress(100, "Done")

    except Exception as e:
        result_data["answers"] = [str(e)]
        set_progress(100, "Error")

# ---------------------------
# Routes
# ---------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/progress")
def get_progress():
    return JSONResponse(progress)

@app.get("/result")
def get_result():
    return JSONResponse(result_data)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    progress["percent"] = 0
    progress["msg"] = "Starting..."

    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())

    threading.Thread(target=process_pdf, args=(path,)).start()
    return JSONResponse({"status": "processing"})
