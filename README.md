# Multimodal AI Assistant

A college mini project that answers questions about **images**, **PDF documents**,
and **spoken questions**. It combines image understanding, visual PDF analysis,
speech recognition, and optional local speech output in one Streamlit app.

> **Provider note:** the project brief mentioned Gemini, but this implementation
> uses the project's existing **Groq** integration. No Gemini SDK, LangChain,
> vector database, or OCR framework is used. All AI requests go through the
> official Groq Python SDK.

---

## 1. Problem Statement

Most student projects can answer questions about plain text, but real study
material is multimodal: lecture slides contain diagrams, screenshots contain
text, PDFs contain tables and scanned pages, and users often prefer speaking
instead of typing. Extracting PDF text alone is not enough because it loses
layout and visual meaning.

This project provides a single assistant that:

- Reads selectable text from images and PDFs.
- Looks at rendered page images so diagrams, tables, charts, and scanned pages
  can also be understood.
- Accepts typed or spoken questions.
- Returns grounded answers with page references where possible.
- Can speak an answer aloud on supported local systems.

## 2. Objectives

1. Integrate image, PDF, and voice question answering in one application.
2. Use a vision-language model for both images and rendered PDF pages.
3. Keep the implementation understandable for a B.Tech mini project.
4. Validate all user inputs and fail with friendly messages.
5. Avoid unnecessary frameworks and permanent storage of user files.
6. Test the processing and AI layers without making paid API calls.

## 3. Main Features

### Image Question Answering

- Upload JPG, JPEG, PNG, or WEBP up to **10 MB**.
- Content is validated with Pillow, not just the filename.
- EXIF orientation is corrected, transparency is placed on white, and large
  images are resized to at most **1600 px** on the long side.
- Ask natural-language questions about objects, scenes, diagrams, or visible text.
- Answers appear as a chat conversation for the current image.

### Multimodal PDF Question Answering

- Upload PDFs up to **20 MB** and **30 pages**.
- Reject empty, corrupted, encrypted, oversized, or over-length PDFs.
- Extract selectable text page by page with correct 1-based page numbers.
- Render pages to PNG at **144 DPI** for visual understanding.
- **Selected page** mode sends one page's text and image.
- **Whole document** mode uses simple keyword scoring to choose at most
  **3 pages**, then sends their text and images.
- Preview the rendered page in the app.
- Show a chat conversation with the pages analyzed for each answer.

### Voice Assistant

- Record from the browser with `st.audio_input` (no PyAudio, no server mic).
- Enforce a **30 second** limit and validate WAV structure and size.
- Transcribe with SpeechRecognition's `recognize_google`.
- Show an editable transcription before submission.
- Route the question explicitly to **General**, **Current Image**, or
  **Current PDF** context.
- Display the answer as chat bubbles.
- Generate a WAV answer with pyttsx3 and play it with `st.audio` where the
  operating system supports local speech synthesis.

### Session Management

- Streamlit `session_state` keeps each browser session separate.
- Answers survive UI reruns.
- Uploading a different image or PDF clears that document's conversation.
- Histories are capped at **20 Q&A pairs**.
- Editing a voice question or changing context invalidates old answers/speech.
- A sidebar **Clear session** button resets all three features.

## 4. Technology Stack

Versions tested in the project virtual environment (**Python 3.11.15**):

| Purpose | Package | Tested version |
| --- | --- | --- |
| Web interface | `streamlit` | 1.66.1 |
| AI provider SDK | `groq` | 1.7.0 |
| PDF parsing and rendering | `PyMuPDF` | 1.28.2 |
| Image validation and resizing | `Pillow` | 13.0.0 |
| Speech-to-text | `SpeechRecognition` | 3.17.0 |
| Local text-to-speech | `pyttsx3` | 2.99 |
| Environment variables | `python-dotenv` | 1.2.4 |
| Testing | `pytest` | 9.2.0 |
| Test HTTP objects | `httpx` | 0.28.1 |

Windows speech support pulled in `comtypes 1.4.17`, `pywin32 312`, and
`pypiwin32 223`. These are installed automatically as pyttsx3 dependencies; they
are not separate project requirements.

**No Google GenAI SDK is installed or required.** The app uses Groq's
OpenAI-compatible chat completions API for both text and vision requests.

## 5. System Architecture

```text
                         Browser (Streamlit UI)
                                  |
        +-------------------------+-------------------------+
        |                         |                         |
   Image upload              PDF upload               Audio recording
        |                         |                   (st.audio_input)
        v                         v                         |
 image_processor.py         pdf_processor.py         voice_processor.py
 - validate                 - validate               - validate WAV
 - orient/resize            - extract text           - SpeechRecognition
 - RGB/re-encode            - render 144 DPI PNG       (Google service)
        |                    - select pages                   |
        |                         |                         v
        +-----------+-------------+-------------------> transcribed text
                    |
                    v
             ai_assistant.py
        - ask_image / ask_pdf / ask_text
        - Groq text / vision model
        - safe error messages
                    |
                    v
             Answer displayed as chat
                    |
                    v
          voice_processor.text_to_speech (pyttsx3)
                    |
                    v
             st.audio browser playback
```

Request flow:

1. The browser sends file or audio bytes to Streamlit.
2. The matching processor validates and prepares the content.
3. `ai_assistant.py` builds a multimodal chat request and sends it to Groq.
4. The answer is stored in `st.session_state` and rendered.
5. Voice answers can be synthesized locally and returned as WAV bytes.

## 6. Folder Structure

```text
Multimodal AI Assistant/
├── app.py                      # Streamlit UI, routing, session state
├── modules/
│   ├── __init__.py
│   ├── ai_assistant.py         # Groq configuration and AI requests
│   ├── image_processor.py      # image validation and preprocessing
│   ├── pdf_processor.py        # PDF validation, text, rendering, selection
│   └── voice_processor.py      # STT, TTS, audio validation, voice routing
├── tests/
│   ├── __init__.py
│   ├── test_ai_assistant.py
│   ├── test_image_processor.py
│   ├── test_pdf_processor.py
│   ├── test_voice_processor.py
│   ├── test_app_integration.py # AppTest session-state tests
│   └── manual_image_qa.py      # optional real-API image check
├── uploads/
│   └── .gitkeep                # uploads are not written here by the app
├── .env.example
├── .gitignore
├── .python-version
├── PROJECT_EXPLANATION.md
├── README.md
└── requirements.txt
```

## 7. Installation

Requirements: **Python 3.11**, internet access for AI/speech services, and a Groq
API key from <https://console.groq.com/keys>.

```bash
python --version
python -m venv .venv
```

Activate the environment:

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install exact pinned dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

## 8. API Key Configuration

Copy the example file:

```powershell
Copy-Item .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
GROQ_API_KEY=your_real_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_VISION_MODEL=qwen/qwen3.8-27b
```

- `GROQ_API_KEY` is required for AI answers and is never displayed.
- `GROQ_MODEL` is used by general text questions (voice General context).
- `GROQ_VISION_MODEL` is used for images and rendered PDF pages.
- `.env` is ignored by Git. Never commit it.

If the key is missing, the app still opens and shows a configuration warning.
Invalid keys, unavailable models, rate limits, timeouts, and empty responses are
reported with short, safe messages.

## 9. How to Run

```bash
streamlit run app.py
```

Open the URL shown in the terminal, normally <http://localhost:8501>.

The app can start without a key. AI features require valid configuration.

## 10. How to Use Each Feature

### Image Question Answering

1. Choose **Image Question Answering** in the sidebar.
2. Upload a PNG, JPEG, or WEBP image.
3. Check the preview and image metadata.
4. Type a question and press **Ask AI**.
5. Read the answer in the conversation below the form.

Example questions:

- "What objects are visible?"
- "Explain this diagram."
- "Read the text in this screenshot."
- "What is unusual about this picture?"

### PDF Question Answering

1. Choose **PDF Question Answering**.
2. Upload a PDF.
3. Review the page count, size, and rendered page preview.
4. Select a preview page.
5. Choose **Selected page** or **Whole document**.
6. Type a question and press **Ask AI**.
7. Read the answer and the pages analyzed.

Example questions:

- "Summarize page 1."
- "Explain the diagram on page 5."
- "Explain the table on page 7."
- "What does this flowchart represent?"
- "What is the difference between paging and segmentation?"
- "Explain this scanned page."

Whole-document mode selects at most three pages using simple keyword overlap.
If no page matches confidently, the app asks you to choose a page manually.

### Voice Assistant

1. Choose **Voice Assistant**.
2. Select **General**, **Current Image**, or **Current PDF**.
   - Current Image/PDF reuse a file already uploaded in that mode.
3. For PDFs, choose selected page or whole document.
4. Press the microphone, ask your question, and stop recording.
5. Review and edit the transcription if needed.
6. Press **Submit Question**.
7. Read the answer and optionally press **Generate Speech**.
8. Use **Reset interaction** to clear the recording and result.

Speech recognition uses Google's online service, so it needs internet access.
Text-to-speech uses the server's local pyttsx3 engine. If no engine is available,
the app keeps the text answer and shows a warning.

## 11. Example Inputs and Outputs

| Input | Question | Expected kind of answer |
| --- | --- | --- |
| Photo of a desk | "What objects are visible?" | List of visible objects grounded in the image |
| Flowchart image | "Explain this diagram." | Description of boxes, arrows, and process flow |
| Text PDF, page 1 | "Summarize page 1." | Summary citing page 1 |
| PDF with a table | "Explain the table on page 7." | Column/row explanation from the rendered page |
| Scanned PDF page | "What does this page contain?" | Visual reading of the rendered scan |
| Spoken general question | "What is artificial intelligence?" | Short text answer, optionally spoken |
| Spoken PDF question | "Summarize this document." | Answer using selected PDF pages |

AI output is probabilistic. Answers should be checked against the source
document, especially for numerical or medical/legal content.

## 12. Testing

```bash
python -m pytest
```

The suite mocks all external AI and speech services, so it does not use your key
or consume credits. Synthetic images, PDFs, and WAV files are created in memory.

Coverage includes:

- Image validation, preprocessing, EXIF orientation, resizing, transparency.
- PDF validation, encrypted/oversized/corrupt files, text extraction, numbering.
- Page rendering, page selection, request construction with text and images.
- Groq text/image/PDF request building and safe API error messages.
- WAV decoding, duration/size limits, mocked speech recognition.
- TTS success, unavailable engine, synthesis failure, text limiting.
- Voice context routing, missing image/PDF context.
- App-level session state, chat history, document reset, clear session, duplicate
  transcription prevention, and missing configuration warning.

Optional real-API image check:

```bash
python -m tests.manual_image_qa --image path/to/image.png
```

This manual script is not collected by pytest because its name does not start
with `test_`.

### Manual demonstration checklist

| # | Check | Status |
| --- | --- | --- |
| 1 | Image: "What objects are visible?" | Requires a real Groq key |
| 2 | Diagram: "Explain this diagram." | Requires a real Groq key |
| 3 | Text PDF: "Summarize page 1." | Requires a real Groq key |
| 4 | Flowchart PDF: "Explain the flowchart." | Requires a real Groq key |
| 5 | Scanned PDF: "What does this page contain?" | Requires a real Groq key |
| 6 | Record "What is artificial intelligence?" | Requires a microphone and Google speech access |
| 7 | Spoken PDF question | Requires a microphone, speech access, and Groq key |
| 8 | Generate speech | Local pyttsx3 synthesis was verified on Windows; audible playback depends on the machine |
| 9 | Remove API key | Automated AppTest verifies the warning |
| 10 | Invalid files | Automated tests verify friendly errors |

## 13. Known Limitations

- **No semantic retrieval.** Whole-document mode uses keyword frequency, not
  embeddings. It can miss a relevant page if the wording differs.
- **At most 3 PDF pages per request**, to stay within the vision model's image
  limit and keep payloads small.
- **Text extraction is not OCR.** Scanned pages are understood from the rendered
  page image, so accuracy depends on scan quality and the vision model.
- Password-protected PDFs are rejected.
- Upload limits: images 10 MB, PDFs 20 MB / 30 pages, audio 30 seconds.
- Speech recognition is online and may fail or mis-transcribe.
- Local TTS needs a working system engine (SAPI5 on Windows, espeak on Linux)
  and is not ideal for a multi-user production server.
- The vision model may not support every diagram or language equally well.
- Answers can be incomplete or wrong; the model is instructed to admit missing
  context but this is not guaranteed.
- The app does not provide authentication, databases, or long-term storage.
- The project uses Groq rather than Gemini; model names must exist in your Groq
  account.

## 14. Security and Privacy

- API keys are read from environment variables; `.env` is git-ignored.
- The UI only reports whether a key is configured.
- File types and sizes are validated before processing.
- Uploaded content is never executed.
- Uploaded files are processed in memory and are not permanently saved by the app.
- TTS temporary WAV files use unique names and are deleted in a `finally` block.
- Request timeouts are set for AI calls.
- Document text, image bytes, and audio are not logged.
- No raw provider error details are shown to users.
- **Uploaded images, PDF pages, and questions may be sent to Groq. Recorded
  audio is sent to Google's speech recognition service.** Do not upload private
  material unless you are comfortable with that processing.

## 15. Future Improvements

- Optional OCR fallback for scanned PDFs.
- Better page retrieval using embeddings, if the project scope allows it.
- Streaming answers for faster perceived response.
- Multilingual speech recognition and voices.
- Cloud TTS for browser-friendly playback on headless servers.
- User accounts and saved history.
- PDF page-range selection and side-by-side citations.
- Support for more document formats.
- Evaluation set for measuring answer accuracy and hallucination rate.

---

## License and Academic Use

This project was developed for academic demonstration. Model names, provider
terms, quotas, and SDK versions should be checked before submission or deployment.
