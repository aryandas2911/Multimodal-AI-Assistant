# Multimodal AI Assistant

A college mini project that combines image question answering, PDF question
answering, and voice-based AI interaction in one Streamlit application.

**Phase 2 is implemented:** users can upload an image, ask a natural-language
question, and receive an answer from a Groq vision model. PDF and voice
features remain planned for later phases.

## Features

### Available now

- **Image Question Answering**
  - Upload JPEG, PNG, or WEBP images up to 10 MB.
  - Preview the uploaded image.
  - Validate the actual file contents (not just the extension).
  - Fix camera orientation using EXIF data.
  - Resize large images to a maximum dimension of 1600 pixels.
  - Ask questions about objects, scenes, diagrams, screenshots, visible text,
    charts, and tables.
  - Keep the latest answer and a short in-session question history.
  - Clear the current image and its answers.

### Planned

- **PDF Question Answering** - query PDF text, images, tables, and diagrams.
- **Voice Assistant** - speak a question and optionally hear a spoken answer.

## Technology Stack

| Purpose | Library |
| --- | --- |
| Web interface | Streamlit 1.66.1 |
| AI API client | Groq Python SDK 1.7.0 |
| PDF processing | PyMuPDF 1.28.2 (not used yet) |
| Image processing | Pillow 13.0.0 |
| Speech-to-text | SpeechRecognition 3.17.0 (not used yet) |
| Local text-to-speech | pyttsx3 2.99 (optional, not used yet) |
| Configuration | python-dotenv 1.2.4 |
| Testing | pytest 9.2.0 |

> **Note:** The original project brief named Gemini. The final instruction was
> to use the Groq API instead, so this project uses the official `groq` SDK and
> `GROQ_API_KEY`.

## Requirements

- **Python 3.11** (recorded in `.python-version`)
- A Groq API key from <https://console.groq.com/keys>
- Internet access for AI requests

The Streamlit app opens without an API key and shows a configuration warning.
Image answers require a valid key.

## Project Structure

```text
multimodal-ai-assistant/
├── app.py
├── modules/
│   ├── __init__.py
│   ├── ai_assistant.py       # config, Groq client, image questions
│   ├── image_processor.py    # validation and preprocessing
│   ├── pdf_processor.py      # planned
│   └── voice_processor.py    # planned
├── tests/
│   ├── __init__.py
│   ├── test_ai_assistant.py
│   ├── test_image_processor.py
│   ├── test_pdf_processor.py
│   ├── test_voice_processor.py
│   └── manual_image_qa.py    # optional real-API check (not run by pytest)
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

You should see Python 3.11.x.

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

Copy the example file:

Windows:

```powershell
Copy-Item .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

Set the values in `.env`:

```dotenv
GROQ_API_KEY=your_real_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_VISION_MODEL=qwen/qwen3.8-27b
```

- `GROQ_API_KEY` is required for AI requests and is never displayed.
- `GROQ_MODEL` is reserved for the later PDF and voice text features.
- `GROQ_VISION_MODEL` is used for image questions. It can be changed without
  editing the code.
- The default vision model, `qwen/qwen3.8-27b`, is Groq's documented multimodal
  model at the time of writing. Check the
  [Groq models page](https://console.groq.com/docs/models) if it changes.

### 5. Run the application

```bash
streamlit run app.py
```

Open <http://localhost:8501>, choose **Image Question Answering**, upload an
image, type a question, and press **Ask AI**.

## Running the Tests

```bash
python -m pytest
```

The unit tests mock the Groq API. They do not use your key or consume credits.

### Optional real-API check

This command makes one real request and may use a small amount of quota:

```bash
python -m tests.manual_image_qa --image path/to/image.png
```

You can ask a specific question:

```bash
python -m tests.manual_image_qa --image diagram.png --question "Explain this diagram."
```

## How Image Processing Works

1. `validate_image()` checks byte size, opens the file with Pillow, verifies
   that the detected format is JPEG, PNG, or WEBP, and decodes the pixels.
2. `get_image_info()` returns the format, dimensions, MIME type, and file size.
3. `prepare_image()` applies EXIF orientation, converts unsupported modes to
   RGB, places transparent areas on a white background, and resizes the longest
   side to at most 1600 pixels.
4. `ask_image()` sends the processed image and question to Groq as one
   multimodal chat request and returns clean answer text.

The original uploaded bytes are not modified.

## Error Handling

The app reports understandable messages for:

- No image uploaded or no question entered.
- Empty, corrupted, oversized, or unsupported images.
- Missing or invalid API key.
- Request timeouts and connection failures.
- Rate limits or exhausted quota.
- An unavailable vision model.
- Rejected requests and empty AI responses.

Raw API keys and provider exception details are not shown in the UI.

## Phase 2 Scope

Implemented:

- Image upload, validation, preview, and preprocessing.
- Groq vision request with a system instruction.
- Answer display, session history, and reset controls.
- Mocked unit tests for image handling, request construction, and API errors.
- Optional manual test for the real API.

Not implemented yet:

- PDF extraction and question answering.
- Speech-to-text and text-to-speech.
- Conversation memory beyond the current image/session.

## Security Notes

- `.env` is ignored by Git.
- The app only reports whether a key is configured; it never prints the key.
- Uploaded files are processed in memory and are not saved by the app.
- `.gitignore` excludes uploaded files and generated audio.
- Do not put API keys in source code or `requirements.txt`.

## Verification

Verified in a Python 3.11.15 virtual environment on Windows:

- `python -m pip check` reported no broken requirements.
- All modules imported successfully.
- `python -m pytest` passed all tests.
- The Streamlit app started and its health endpoint returned `200 ok`.
