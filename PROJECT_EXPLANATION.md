# Project Explanation

A plain-language guide to the Multimodal AI Assistant for demonstrations, report
writing, and viva preparation.

---

## 1. Simple Project Introduction

The Multimodal AI Assistant is a Streamlit web application that lets a user ask
questions about:

1. **An image** — photos, screenshots, charts, or diagrams.
2. **A PDF** — selectable text, tables, diagrams, charts, and scanned pages.
3. **A spoken question** — recorded in the browser, transcribed to text, and
   answered by the AI.

The answer is displayed as text. In the voice mode, it can also be converted into
speech and played back in the browser.

The project combines four ideas: a web interface, file processing, a
vision-language model, and speech processing. It is deliberately small enough to
understand as a college mini project and does not use LangChain, vector
databases, agents, or cloud infrastructure.

> The project brief used the word "Gemini", but the implementation uses the
> existing **Groq** API integration. The architecture is otherwise the same: a
> multimodal chat request carrying text and images to a hosted model.

## 2. Why the Project Is Useful

- **Study material is visual.** Text extraction alone loses flowcharts, tables,
  labels, and scanned notes.
- **Asking is faster than searching.** A student can ask a direct question
  instead of reading a long PDF.
- **Voice is convenient.** A spoken question is useful for demonstrations and
  accessibility.
- **It teaches integration.** The project connects file validation, PDF
  rendering, API calls, speech services, session state, and testing in one app.
- **It is honest about limits.** Keyword page selection, online speech
  recognition, and model hallucination are documented instead of hidden.

## 3. How Image Question Answering Works

1. The user uploads a JPG, PNG, or WEBP file.
2. `image_processor.py` checks the byte size and opens the file with Pillow.
3. Pillow detects the real format from the file signature. Renaming a text file
   to `.png` does not bypass this check.
4. The image is rotated using EXIF data, converted to RGB, transparent areas are
   placed on white, and the image is resized to at most 1600 px.
5. The prepared image is base64-encoded and placed in a multimodal chat message.
6. The question text and image are sent to the Groq vision model.
7. The answer is stored in `st.session_state` and displayed in a chat bubble.

The original upload is previewed in the UI. The model receives the resized copy,
which reduces payload size while keeping text and diagrams reasonably readable.

## 4. How PDF Text Extraction Works

1. `pdf_processor.validate_pdf()` opens the bytes with
   `pymupdf.open(stream=pdf_bytes, filetype="pdf")`.
2. It rejects empty, corrupted, encrypted/password-protected, oversized
   (>20 MB), and too-long (>30 pages) PDFs.
3. `extract_pdf_text()` visits every page and calls `page.get_text("text")`.
4. Whitespace is normalized: blank lines are removed and repeated spaces are
   collapsed.
5. Each page becomes a dictionary with a **1-based** page number, text,
   `has_text` flag, and character count.
6. Pages with no text layer are marked `has_text=False`. These are usually
   scanned/image-only pages.

This extraction only reads selectable text. It does not perform OCR.

## 5. How PDF Diagrams Are Understood

Text extraction is not enough for diagrams, tables, charts, screenshots, or
scanned pages. Therefore the app also renders each selected page as a PNG:

```python
pixmap = page.get_pixmap(dpi=144, alpha=False)
png_bytes = pixmap.tobytes("png")
```

The rendered image preserves layout, arrows, labels, table lines, and visual
content. `ask_pdf()` sends **both** the extracted text and the rendered page image
for each supplied page. The model is instructed to inspect the image and cite
page numbers.

Rendering is capped at about 12 megapixels to avoid excessive memory use. Page
rendering is cached by Streamlit using the PDF bytes and page number, so the same
page is not rendered repeatedly.

## 6. How Speech Recognition Works

1. `st.audio_input` records audio in the browser and sends WAV bytes to the
   Streamlit server. No server microphone or PyAudio is used.
2. The WAV bytes are validated with Python's standard `wave` module: size,
   sample rate, channels, sample width, and duration (maximum 30 seconds).
3. `speech_to_text()` opens the bytes with `sr.AudioFile(BytesIO(...))`.
4. `recognizer.record(source)` converts the WAV into SpeechRecognition
   `AudioData`.
5. `recognizer.recognize_google(audio_data)` sends the audio to Google's online
   speech service and returns text.
6. The transcript is displayed in an editable text area.
7. The audio hash (SHA-256) is stored. If Streamlit reruns with the same
   recording, the saved transcript is reused instead of contacting the service
   again.

Speech recognition is **online**, not offline, and is not guaranteed to be
accurate or available.

## 7. How Text-to-Speech Works

1. The user presses **Generate Speech** for the current answer.
2. `text_to_speech()` cleans the text and limits it to 1200 characters for speech
   (the full text answer remains unchanged).
3. pyttsx3 initializes the operating system's speech engine (SAPI5 on Windows,
   often espeak on Linux).
4. The engine writes a WAV file to a unique temporary path created with
   `tempfile.mkstemp`.
5. The file is checked for a valid RIFF/WAVE header and read into bytes.
6. The temporary file is deleted in a `finally` block, and the engine is stopped.
7. `st.audio()` plays the WAV bytes in the browser.

If no speech engine exists or synthesis fails, a friendly warning is shown and
the text answer remains available. pyttsx3 is not ideal for a shared production
server because it depends on local operating-system audio support.

## 8. How the AI Processes Multimodal Input

The AI layer is in `modules/ai_assistant.py`.

- `ask_text()` sends a normal text-only message using `GROQ_MODEL`.
- `ask_image()` sends one text part and one base64 image part using
  `GROQ_VISION_MODEL`.
- `ask_pdf()` sends page text blocks with page markers, followed by one rendered
  image per page, using `GROQ_VISION_MODEL`.

Groq exposes an OpenAI-compatible chat completions API. A multimodal message
looks conceptually like:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Question and PDF page text..."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
  ]
}
```

The model combines the textual question, extracted document text, and visual
page images to produce an answer. The prompt asks it to use only the supplied
context, admit missing information, and cite page numbers where possible.

Request limits in the project:

- Maximum 3 PDF pages per request.
- Maximum 30,000 characters of PDF text.
- Maximum 8 MB of rendered page images.
- 60-second request timeout.
- Low temperature (0.2) for more consistent answers.

## 9. Explanation of Each Python Module

### `app.py`

The Streamlit entry point. It:

- Configures the page and sidebar.
- Switches between the three modes.
- Handles uploads and widgets.
- Stores documents, transcripts, and chat history in `st.session_state`.
- Calls the processing and AI modules.
- Displays previews, spinners, errors, chat bubbles, and audio players.
- Clears document-specific state when a new file is uploaded.

### `modules/image_processor.py`

- `validate_image()` checks bytes, format, dimensions, and decodability.
- `get_image_info()` returns format, MIME type, width, height, and size.
- `prepare_image()` applies EXIF orientation, converts to RGB, flattens
  transparency, resizes, and re-encodes for the model.
- Constants define the 10 MB limit, 40 MP safety limit, and 1600 px resize limit.

### `modules/pdf_processor.py`

- `validate_pdf()` opens and checks the PDF, then closes it.
- `get_pdf_info()` returns page count, size, title, and MIME type.
- `extract_pdf_text()` returns page-by-page normalized text.
- `render_pdf_page()` renders one 1-based page to PNG at 144 DPI.
- `select_relevant_pages()` implements explicit page references and simple
  keyword scoring for whole-document questions.
- `PDFProcessingError` carries user-safe messages.

### `modules/voice_processor.py`

- `get_audio_info()` / `validate_audio()` parse and validate WAV data.
- `audio_signature()` creates a SHA-256 identity for a recording.
- `speech_to_text()` converts WAV bytes to text with SpeechRecognition.
- `transcribe_audio()` reuses a cached transcript for unchanged audio.
- `resolve_voice_context()` checks that the selected context has a file.
- `text_to_speech()` synthesizes a WAV with pyttsx3 and cleans up temporary
  files.
- `VoiceProcessingError` carries user-safe messages.

### `modules/ai_assistant.py`

- `get_settings()` reads and validates environment variables.
- `create_client()` constructs the Groq client without a network request.
- `ask_text()`, `ask_image()`, and `ask_pdf()` build requests.
- `_request_completion()` centralizes timeouts, token limits, and conversion of
  provider errors into safe messages.
- `AppSettings` hides the API key from its `repr`.

### Tests

- `test_image_processor.py` — validation and preprocessing.
- `test_pdf_processor.py` — PDF validation, text, rendering, selection, requests.
- `test_voice_processor.py` — WAV validation, STT, TTS, routing, deduplication.
- `test_ai_assistant.py` — configuration, text/image requests, API errors.
- `test_app_integration.py` — AppTest checks for session state and UI wiring.
- `manual_image_qa.py` — optional real-API image check.

## 10. Overall Data Flow

```text
User input
   |
   +-- Image bytes --> validate -> prepare -> base64 -> ask_image -----+
   |                                                                   |
   +-- PDF bytes ----> validate -> extract text -> render pages ------+
   |                              -> select pages -> ask_pdf ----------+
   |                                                                   |
   +-- Audio bytes --> validate -> speech_to_text -> editable text ----+
                                                                       |
                                                                       v
                                                        Groq model request
                                                                       |
                                                                       v
                                                              answer text
                                                                       |
                                        +------------------------------+---+
                                        |                                  |
                                        v                                  v
                              chat_message display                pyttsx3 WAV
                                                                           |
                                                                           v
                                                                     st.audio
```

Session state controls lifetime:

- New image hash → clear image conversation and old image bytes.
- New PDF hash → clear PDF conversation, page selection, and render cache.
- New audio hash → transcribe once; new answer invalidates old speech.
- Sidebar **Clear session** → reset everything.

## 11. Important Technical Decisions

1. **Groq instead of Gemini.** The project already had a working Groq
   integration. Adding a second AI SDK would increase dependencies without
   improving the architecture.
2. **Render PDF pages, do not rely on text alone.** This is the key difference
   between plain text extraction and multimodal document understanding.
3. **1-based page numbers.** Users and prompts think in pages 1, 2, 3, while
   PyMuPDF uses 0-based indexes internally.
4. **Selected-page mode as the reliable path.** Keyword scoring is only a
   heuristic, so users can always choose a page directly.
5. **At most 3 pages per request.** This respects the vision model's image limit
   and keeps requests small.
6. **Content hashing for identity.** SHA-256 identifies images, PDFs, audio, and
   answers without comparing entire byte arrays repeatedly.
7. **Streamlit caching only for deterministic work.** Rendered PDF pages are
   cached; AI calls and speech calls are not hidden behind caches.
8. **Forms for questions.** `st.form` prevents an API call on every widget rerun.
9. **Safe error messages.** Provider exceptions are converted into short
   messages with `from None` so raw details and secrets do not reach the UI.
10. **In-memory processing.** Uploaded files are held in memory and are not
    written permanently to disk. TTS uses a unique temporary file that is deleted.
11. **No PyAudio.** Browser recording avoids server microphone hardware.
12. **No OCR framework.** Scanned pages are read visually from rendered images.

## 12. Viva Questions and Answers

### Q1. What is multimodal AI?

Multimodal AI can process more than one type of input, such as text and images,
and combine them to produce a response. This project sends question text together
with image or PDF-page images to a vision-language model.

### Q2. What is an LLM?

A large language model is a neural network trained on large text corpora to
predict and generate text. It processes token sequences and produces responses,
but by itself it does not directly see images.

### Q3. What is a vision-language model?

A vision-language model accepts images as well as text. An image encoder turns
the picture into representations that the language-model component can use to
answer questions about visual content.

### Q4. How is this project multimodal?

It accepts image files, PDF text and page images, and audio recordings. Audio is
converted to text, then the text and image/page context are sent to the AI. The
answer can be returned as text and speech.

### Q5. What is the difference between OCR and visual document understanding?

OCR detects and converts characters in an image into machine-readable text.
Visual document understanding uses a model to interpret layout, tables, diagrams,
arrows, and relationships in addition to text. This project has no OCR engine; it
renders pages and lets the vision model read them visually.

### Q6. Why not rely only on PDF text extraction?

Because extracted text loses visual structure. A flowchart's arrows, a table's
rows and columns, a chart's axes, and a scanned page may have no useful text
layer. Sending rendered pages preserves that information.

### Q7. How does PyMuPDF extract text?

`page.get_text("text")` reads the PDF's text layer in normal reading order and
returns a string. Pages without a text layer return little or no text.

### Q8. Why are PDF pages rendered at 144 DPI?

144 DPI gives readable text and diagrams without producing very large images.
It is within the recommended 120–150 DPI range for this project. A pixel cap
prevents unusually large pages from using too much memory.

### Q9. Why use 1-based page numbers?

Users refer to "page 1" as the first page. PyMuPDF uses 0-based indexes, so the
project converts at the boundary (`page_number - 1`) and exposes 1-based numbers
everywhere else.

### Q10. How does whole-document page selection work?

The question is lowercased and split into words. Stopwords are removed, and each
page receives a score equal to how often the remaining keywords occur. The highest
scoring pages are selected, up to three. Explicit "page N" references are always
used. This is keyword frequency, not semantic retrieval.

### Q11. Why is keyword matching not semantic search?

It matches surface words only. If the question says "memory management" but the
page says "virtual storage", the keyword score may be zero even though the page is
relevant. Embeddings would capture meaning, but that is outside this project's
scope.

### Q12. Why limit PDF requests to three pages?

The vision model accepts a limited number of images per request, and more images
increase payload size, latency, and cost. Three pages keeps requests predictable.
Users can switch to selected-page mode for precision.

### Q13. How does speech recognition work here?

The browser records WAV audio. SpeechRecognition reads it with `sr.AudioFile` and
`record()`. `recognize_google()` sends the audio to Google's online service and
returns a transcript.

### Q14. Why not use `speech_recognition.Microphone()`?

`Microphone()` requires PyAudio and access to the machine running Python. In a web
app, the microphone belongs to the user's browser, not the server. `st.audio_input`
correctly captures audio on the client side.

### Q15. Is speech recognition offline?

No. `recognize_google` uses an online Google service. It needs internet access and
may fail or return inaccurate text. The app therefore lets the user edit the
transcript before submitting it.

### Q16. How does text-to-speech work?

pyttsx3 uses the operating system's speech engine to synthesize a WAV file. The
app reads the WAV bytes, deletes the temporary file, and plays the bytes with
`st.audio`. If no engine is available, only the text answer is shown.

### Q17. Why can't pyttsx3 speak directly in the user's browser?

pyttsx3 runs on the server. It can produce an audio file, but it cannot access the
remote browser's speakers. The browser must receive bytes through `st.audio`.

### Q18. What is Streamlit session state?

`st.session_state` is per-browser-session storage that survives script reruns. It
stores uploads, transcripts, answers, and histories. Separate users get separate
sessions.

### Q19. Why do Streamlit reruns matter?

Every widget interaction reruns the script. Without session state, answers would
disappear. Without forms, API calls could fire on every rerun. Without audio
hashing, transcription could repeat unnecessarily.

### Q20. How are uploaded files validated?

Images are decoded with Pillow and checked for format, dimensions, and corruption.
PDFs are opened with PyMuPDF and checked for encryption, page count, and validity.
Audio is parsed with the `wave` module and checked for duration and structure.
Byte-size limits are checked first.

### Q21. Why hash the uploaded file?

A SHA-256 hash gives a stable identity. When the hash changes, the app knows the
document changed and clears old answers. It also prevents retranscribing unchanged
audio and reusing stale speech.

### Q22. What is model hallucination?

Hallucination is when a model produces plausible but unsupported or incorrect
content. The prompts instruct the model to use only the supplied context and admit
missing information, but this reduces rather than eliminates the risk.

### Q23. How does the project reduce hallucination?

- It supplies actual document text and images.
- It asks the model to cite pages.
- It tells the model to say when context is insufficient.
- It limits the supplied context instead of silently shortening it.
- It shows which pages were analyzed so the user can verify.

### Q24. What API limitations affect the project?

Rate limits, quotas, request timeouts, image-count limits, token limits, model
availability, and network failures. The app converts these into friendly messages
and uses timeouts and bounded payloads.

### Q25. Why is the temperature set to 0.2?

Lower temperature makes responses more deterministic and factual, which is
preferable for document question answering. It is not a guarantee of correctness.

### Q26. What happens when the API key is missing?

`get_settings()` raises `ConfigurationError`. The sidebar shows a warning, and AI
actions show "AI is not configured". The app still opens, so the UI can be
demonstrated without a key.

### Q27. Are uploaded files stored permanently?

No. The app processes files in memory. It does not write them to the `uploads`
folder. Speech synthesis uses a unique temporary file that is deleted in a
`finally` block. However, content is sent to external Groq and Google services.

### Q28. What data leaves the local machine?

Images, PDF page text, rendered pages, and questions are sent to Groq. Recorded
audio is sent to Google's speech recognition service. API keys are sent only as
authentication to Groq.

### Q29. Why pin dependency versions?

Pinned versions make installations reproducible. Without pins, a future release
could change behavior or break compatibility. `pip check` verifies that installed
packages' requirements are satisfied.

### Q30. How are external services mocked in tests?

Tests replace the Groq client with a fake object that records requests and returns
a prepared response. SpeechRecognition uses a fake recognizer, and pyttsx3 uses a
fake engine. This makes tests fast, deterministic, and free.

### Q31. Why use `st.chat_message`?

It provides a native, accessible chat layout that clearly separates user questions
from assistant answers without custom CSS or JavaScript.

### Q32. What are the project's main limitations?

Keyword-based page retrieval, no OCR engine, an online speech service, local TTS
availability, model hallucination, file-size/page limits, and no authentication or
persistent storage.

### Q33. What would you improve next?

Add optional OCR, semantic page retrieval, streaming answers, multilingual speech,
cloud TTS, saved sessions, and an evaluation set for measuring answer accuracy.

---

## Quick Demonstration Script

1. Open the app and point out the sidebar configuration status.
2. Upload an image and ask "What objects are visible?"
3. Upload a PDF, preview page 1, and ask "Summarize page 1."
4. Switch to a diagram page and ask "Explain this diagram."
5. Open the voice mode, record "What is artificial intelligence?", edit the
   transcription, and submit.
6. Press **Generate Speech** and play the answer.
7. Clear the session and explain how state isolation works.
