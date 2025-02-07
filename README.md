```
# AI-powered Question Generation API

This project is a FastAPI-based backend that generates questions from a PDF and validates user answers using an AI model.

## 🚀 Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. **Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## 🔥 Running the API

Start the FastAPI server:
```sh
uvicorn home:app --reload
```

## 📂 Endpoints
### `/generate_questions`
- **Method:** `POST`
- **Params:** `file: PDF`, `difficulty: str`
- **Response:** Generates AI-powered questions from a PDF file.

### `/validate_answers`
- **Method:** `POST`
- **Params:** `file: PDF`, `question: str`, `user_answer: str`
- **Response:** Validates the user's answer against the extracted context from the uploaded PDF.

## ✅ Deployment
To deploy this API, use a cloud service like AWS, Azure, or deploy locally using Docker.

## 📌 Notes
- Ensure `transformers` and `PyMuPDF` dependencies are installed.
- This repository ignores large files using `.gitignore`.
