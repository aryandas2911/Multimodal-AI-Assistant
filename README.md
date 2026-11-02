# Multimodal AI Assistant

A college mini project that combines image question answering, multimodal PDF
question answering, and (later) voice-based AI interaction in one Streamlit
application.

**Phase 3 is implemented:** users can upload a PDF, preview a rendered page, and
ask questions about selectable text, diagrams, tables, charts, and scanned
pages. The AI receives both extracted text and rendered page images.

## Features

### Available now

- **Image Question Answering**
  - Upload JPEG, PNG, or WEBP images up to 10 MB.
  - Validate file contents, correct EXIF orientation, and resize large images.
  - Ask about objects, scenes, diagrams, screenshots, and visible text.

- **PDF Question Answering**
  - Upload PDFs up to 20 MB and 30 pages.
  - Validate, open, and reject empty, corrupted, encrypted, or oversized PDFs.
  - Extract selectable text page by page while preserving page numbers.
  - Render pages as PNG at 144 DPI for visual understanding.
  - **Selected page mode:** send one page's text and rendered image.
  - **Whole document mode:** use keyword scoring to select up to 3 pages, then
    send their text and images.
  - Preview the selected page in the application.
  - Keep the latest answer and a session-only question history.
  - Show which pages were analyzed and ask the model to cite page numbers.
  - Clear the current document and its answers.

### Planned

- **Voice Assistant** - speak a question and optionally hear a spoken answer.

## Technology Stack

| Purpose | Library |
| --- | --- |
| Web interface | Streamlit 1.66.1 |
| AI API client | Groq Python SDK 1.7.0 |
| PDF processing | PyMuPDF 1.28.2 |
| Image processing | Pillow 13.0.0 |
| Speech-to-text | SpeechRecognition 3.17.0 (not used yet) |
| Local text-to-speech | pyttsx3 2.99 (optional, not used yet) |
| Configuration | python-dotenv 1.2.4 |
| Testing | pytest 9.2.0, httpx 0.28.1 |

> **Note:** The project brief named Gemini, but the project currently uses the
> existing Groq integration. No Gemini SDK or new AI dependency was added.
> The same vision-capable Groq model handles images and rendered PDF pages.

## Requirements

- **Python 3.11** (recorded in `.python-version`)
- A Groq API key from <https://console.groq.com/keys>
- Internet access for AI requests

The application opens without an API key and shows a configuration warning.
AI answers require a valid key.

## Project Structure

```text
multimodal-ai-assistant/
├── app.py
├── modules/
│   ├── __init__.py
│   ├── ai_assistant.py       # configuration and Groq AI requests
│   ├── image_processor.py    # image validation and preprocessing
│   ├── pdf_processor.py      # PDF validation, text, rendering, page selection
│   └── voice_processor.py    # planned
├── tests/
│   ├── __init__.py
│   ├── test_ai_assistant.py
│   ├── test_image_processor.py
│   ├── test_pdf_processor.py
│   ├── test_voice_processor.py
│   └── manual_image_qa.py    # optional real-API check
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
- `GROQ_MODEL` is reserved for future text-only features.
- `GROQ_VISION_MODEL` handles both image questions and rendered PDF pages.
- The default vision model is configured in `modules/ai_assistant.py` and can
  be overridden in `.env`.

### 5. Run the application

```bash
streamlit run app.py
```

Open <http://localhost:8501>.

## Using PDF Question Answering

1. Choose **PDF Question Answering** in the sidebar.
2. Upload a PDF.
3. Review the page count, file size, and rendered page preview.
4. Select **Selected page** or **Whole document**.
5. Type a question and press **Ask AI**.
6. Read the answer and the list of pages that were analyzed.

Example questions:

- "Summarize page 2."
- "Explain the diagram on page 5."
- "What is the difference between paging and segmentation?"
- "Explain the table on page 7."
- "What does this flowchart represent?"
- "List the important topics discussed in this document."
- "Explain this scanned page."

The page images are important. A scanned page has no selectable text, and a
diagram or table may lose meaning if only its text is extracted. The AI prompt
therefore includes rendered page images for every supplied page.

## How PDF Processing Works

1. `validate_pdf()` opens the uploaded bytes with
   `pymupdf.open(stream=pdf_bytes, filetype="pdf")`, rejects encrypted,
   corrupted, empty, oversized, or too-long documents, and closes the handle.
2. `extract_pdf_text()` reads selectable text page by page, normalizes
   whitespace, and marks image-only pages with `has_text=False`.
3. `render_pdf_page()` renders a 1-based page number to PNG at 144 DPI and
   validates the page boundaries.
4. `select_relevant_pages()` handles whole-document mode:
   - explicit references such as "page 4" are used first;
   - remaining pages are ranked by simple keyword overlap;
   - at most 3 pages are selected;
   - if no page matches confidently, the user is asked to select one.
5. `ask_pdf()` sends the selected pages' text and rendered images in one
   multimodal Groq request and asks for page citations.

Page rendering is cached by Streamlit using the PDF content and page number.
The cache is cleared when a different document is loaded.

## Running the Tests

```bash
python -m pytest
```

Automated tests mock the Groq API. They do not use your key or consume credits.
Synthetic PDFs are generated with PyMuPDF, so no external PDF files are needed.

Tests cover:

- Valid, empty, invalid, encrypted, oversized, and over-length PDFs.
- Multi-page text extraction and page numbering.
- Whitespace normalization and image-only pages.
- Page rendering and invalid page numbers.
- Keyword-based page selection and manual-selection fallback.
- PDF request construction with both text and page images.
- Safe handling of API errors.

### Optional real-API check

The image-only manual script is still available and makes one real request:

```bash
python -m tests.manual_image_qa --image path/to/image.png
```

To test PDFs against the real API, use the Streamlit UI and ask a question
about a small PDF.

## Phase 3 Scope

Implemented:

- PDF upload, validation, page counting, and text extraction.
- 144 DPI page rendering and in-app preview.
- Selected-page and whole-document question modes.
- Keyword-based page selection with a clear confidence limitation.
- Text plus rendered-image multimodal PDF requests.
- Session history, latest answer, page references, and clear controls.
- Automated tests for processing, rendering, selection, and AI request building.

Not implemented yet:

- Voice interaction.
- Semantic search, embeddings, vector databases, or OCR engines.
- Password-protected PDFs.
- Guaranteed retrieval of every relevant page in a long document.

## Limitations

- Whole-document mode uses **simple keyword matching**, not semantic retrieval.
  It can miss a relevant page if the question uses different wording.
- At most 3 pages are sent per request to stay within model and payload limits.
- For long or ambiguous documents, use **Selected page** mode.
- Text extraction is not OCR. Scanned pages are understood from the rendered
  page image, so accuracy depends on image clarity and the vision model.
- Very large pages that would exceed the rendering memory cap are rejected.
- PDFs must be 20 MB or smaller and no more than 30 pages.

## Security Notes

- `.env` is ignored by Git.
- The app only reports whether a key is configured; it never prints the key.
- Uploaded PDFs are processed in memory and are not saved by the app.
- `.gitignore` excludes uploaded files and generated audio.
- Extracted PDF text is not logged.
- Do not put API keys in source code or `requirements.txt`.
