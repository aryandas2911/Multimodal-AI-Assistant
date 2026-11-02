# Multimodal AI Assistant

A college mini project that will combine image question answering, PDF
question answering, and voice-based AI interaction in one Streamlit
application.

This repository currently contains **Phase 1** only: project setup,
configuration, dependency management, tests for the configuration layer, and a
runnable Streamlit shell. The AI features themselves are intentionally not
implemented yet.

## Planned Features

- **Image Question Answering** — upload an image and ask questions about it.
- **PDF Question Answering** — query PDF text, images, tables, and diagrams.
- **Voice Assistant** — speak a question and optionally hear a spoken answer.

## Technology Stack

| Purpose | Library |
| --- | --- |
| Web interface | Streamlit |
| AI API client | Groq Python SDK (`groq`) |
| PDF processing | PyMuPDF (`pymupdf`) |
| Image processing | Pillow |
| Speech-to-text | SpeechRecognition |
| Local text-to-speech | pyttsx3 (optional) |
| Configuration | python-dotenv |
| Testing | pytest |

> **Note:** The original project brief named Gemini. The final instruction was
> to use the Groq API instead, so this Phase 1 uses the official `groq` SDK and
> `GROQ_API_KEY`. The rest of the architecture is unchanged.

## Requirements

- **Python 3.11** (the project baseline; recorded in `.python-version`)
- A Groq API key from <https://console.groq.com/keys> for later AI features

Phase 1 runs without an API key. The app starts and shows a configuration
warning, but AI features will require a key in later phases.

## Project Structure

```text
multimodal-ai-assistant/
├── app.py
├── modules/
│   ├── __init__.py
│   ├── ai_assistant.py
│   ├── image_processor.py
│   ├── pdf_processor.py
│   └── voice_processor.py
├── tests/
│   ├── __init__.py
│   ├── test_ai_assistant.py
│   ├── test_image_processor.py
│   ├── test_pdf_processor.py
│   └── test_voice_processor.py
├── uploads/
│   └── .gitkeep
├── .env.example
├── .gitignore
├── .python-version
├── requirements.txt
└── README.md
```

## Setup

### 1. Check Python

```bash
python --version
```

You should see Python 3.11.x. If not, install Python 3.11 and create the
environment with that interpreter.

### 2. Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Windows (Command Prompt):

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Configure the environment

Copy the example file and edit it:

Windows:

```powershell
Copy-Item .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

Open `.env` and set:

```dotenv
GROQ_API_KEY=your_real_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

- `GROQ_API_KEY` is required for AI calls in later phases. Never commit `.env`.
- `GROQ_MODEL` is configurable. Model availability should be checked against
  the Groq models endpoint when AI integration is implemented.
- The API key is never sent to the browser or displayed in the app.

### 5. Run the application

```bash
streamlit run app.py
```

Streamlit opens <http://localhost:8501> in your browser.

## Running the Tests

```bash
python -m pytest
```

## Phase 1 Scope

Implemented:

- Project directory structure.
- Pinned dependencies.
- Environment-based configuration with clear errors.
- Groq client creation without making a network request.
- Streamlit shell with the three selectable modes and a placeholder panel.
- Tests for configuration validation.

Not implemented yet:

- Image question answering.
- PDF extraction and question answering.
- Speech-to-text and text-to-speech.
- Any Groq API call.

## Phase 1 Verification

The pinned dependency set was installed in a clean Python 3.11.15 virtual
environment on Windows. `python -m pip check` reported no broken requirements,
all modules imported successfully, and the test suite passed. The Streamlit
server was started and its health endpoint returned `200 ok`.

## Security Notes

- `.env` is ignored by Git.
- The application only reports whether a key is configured; it never prints
  the key.
- Uploaded files and generated audio are ignored by Git.
- Do not put API keys directly in source code or `requirements.txt`.
