from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF for extracting text
from transformers import pipeline
from typing import List, Dict
import random
import difflib
import re
import logging

app = FastAPI(title="AI-powered Question Generation and Answer Validation API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the AI model for question generation
generator = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

# Load the AI model for answer validation
qa_pipeline = pipeline("question-answering")

ANSWER_TYPES = ["MCQ", "Sí/No", "Respuesta corta"]


async def extract_text_from_pdf(pdf_file: UploadFile) -> str:
    """Extract text from an uploaded PDF file asynchronously with improved error handling."""
    try:
        if not pdf_file.filename.endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF.")
        
        pdf_bytes = await pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        if len(doc) == 0:
            raise HTTPException(status_code=400, detail="PDF has no pages.")
        
        text = "\n".join([page.get_text("text") for page in doc])
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No extractable text found in PDF.")
        
        return text[:5000].strip()  # Limit text to avoid memory issues
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")


def clean_generated_text(text: str) -> str:
    """Cleans up generated text to remove invalid or nonsensical sentences."""
    text = text.strip()
    return text if text.endswith("?") else ""


def generate_answer_type() -> str:
    """Randomly selects an answer type: MCQ, Sí/No, or Respuesta corta."""
    return random.choice(ANSWER_TYPES)


def generate_possible_answers(question: str, answer_type: str) -> List[str]:
    """Generates possible answers based on the selected answer type."""
    try:
        if answer_type == "MCQ":
            return [
                generator(f"Provide a correct answer to: {question}", max_length=50, num_return_sequences=1, do_sample=True)[0]["generated_text"],
                generator(f"Provide a wrong answer to: {question}", max_length=50, num_return_sequences=1, do_sample=True)[0]["generated_text"],
                generator(f"Provide another wrong answer to: {question}", max_length=50, num_return_sequences=1, do_sample=True)[0]["generated_text"],
                generator(f"Provide one more wrong answer to: {question}", max_length=50, num_return_sequences=1, do_sample=True)[0]["generated_text"]
            ]
        elif answer_type == "Sí/No":
            return ["Sí", "No"]
        elif answer_type == "Respuesta corta":
            return [generator(f"Provide a short answer to: {question}", max_length=50, num_return_sequences=1, do_sample=True)[0]["generated_text"]]
    except Exception:
        return ["No se pudo generar respuesta"]
    return []


@app.post("/generate_questions", response_model=Dict[str, List[Dict[str, object]]])
async def generate_questions(file: UploadFile, difficulty: str = Form("medium")):
    """Receives a PDF file, extracts text, and generates structured questions with AI."""
    text = await extract_text_from_pdf(file)
    if not text:
        raise HTTPException(status_code=400, detail="No text extracted from PDF.")
    
    # Splitting text into smaller chunks to improve question quality
    chunks = [text[i:i+500] for i in range(0, len(text), 500)][:5]  # Use up to 5 chunks
    
    questions = []
    for chunk in chunks:
        prompt = f"Generate a clear and structured question based on the following text with {difficulty} difficulty: {chunk}"
        try:
            generated = generator(prompt, max_length=128, num_return_sequences=2, do_sample=True)
            for q in generated:
                clean_q = clean_generated_text(q["generated_text"])
                if clean_q:
                    answer_type = generate_answer_type()
                    possible_answers = generate_possible_answers(clean_q, answer_type)
                    questions.append({"question": [clean_q], "type": [answer_type], "answers": possible_answers})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"AI generation error: {str(e)}")
    
    if not questions:
        raise HTTPException(status_code=500, detail="Failed to generate valid questions.")
    
    return {"questions": questions[:10]}  # Return up to 10 questions with answers

def clean_text(text: str) -> str:
    """Normalize text by removing punctuation, extra spaces, and converting to lowercase."""
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text.lower()).strip()
    return text
def is_answer_correct(user_answer: str, correct_answer: str, threshold: float = 0.8) -> bool:
    """Check if user answer is similar to correct answer based on a similarity threshold."""
    user_answer_clean = clean_text(user_answer)
    correct_answer_clean = clean_text(correct_answer)
    similarity = difflib.SequenceMatcher(None, user_answer_clean, correct_answer_clean).ratio()
    return similarity >= threshold

@app.post("/validate_answers", response_model=Dict[str, str])
async def validate_answers(file: UploadFile, question: str, user_answer: str):
    """Validates the user's answer using an AI model by extracting text from a newly uploaded PDF."""
    context = await extract_text_from_pdf(file)
    
    if not question or not user_answer or not context:
        raise HTTPException(status_code=400, detail="Invalid input: question, answer, and PDF file are required.")
    
    try:
        response = qa_pipeline(question=question, context=context)
        correct_answer = response.get("answer", "")
        
        if not correct_answer:
            logger.warning(f"No valid answer found for question: {question}")
            raise HTTPException(status_code=400, detail="No valid answer found in the context.")
        
        is_correct = "yes" if is_answer_correct(user_answer, correct_answer) else "no"
    except Exception as e:
        logger.error(f"AI validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI validation error: {str(e)}")
    
    return {"question": question, "user_answer": user_answer, "correct_answer": correct_answer, "is_correct": is_correct}



@app.get("/")
def home():
    return {"message": "Welcome to the AI-powered Question Generation and Answer Validation API!"}
