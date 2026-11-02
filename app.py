"""Streamlit entry point for the Multimodal AI Assistant (Phase 3)."""

import hashlib

import streamlit as st

from modules.ai_assistant import (
    AIAssistantError,
    ConfigurationError,
    MAX_QUESTION_LENGTH,
    ask_image,
    ask_pdf,
    ask_text,
    get_settings,
)
from modules.image_processor import (
    MAX_IMAGE_BYTES,
    ImageValidationError,
    get_image_info,
    prepare_image,
    validate_image,
)
from modules.pdf_processor import (
    MAX_PDF_BYTES,
    MAX_PDF_PAGES,
    RENDER_DPI,
    PDFProcessingError,
    extract_pdf_text,
    get_pdf_info,
    render_pdf_page,
    select_relevant_pages,
    validate_pdf,
)
from modules.voice_processor import (
    MAX_AUDIO_SECONDS,
    VOICE_CONTEXTS,
    VoiceProcessingError,
    audio_signature,
    get_audio_info,
    resolve_voice_context,
    text_to_speech,
    transcribe_audio,
)

# Modes are defined once and reused by both the sidebar and the page content.
MODES = {
    "Image Question Answering": "Ask questions about an uploaded image.",
    "PDF Question Answering": (
        "Ask questions about PDF text, images, tables, diagrams, and scanned pages."
    ),
    "Voice Assistant": (
        "Record a spoken question, review the transcription, and get a text "
        "or spoken answer."
    ),
}

IMAGE_FILE_TYPES = ["jpg", "jpeg", "png", "webp"]

# Streamlit requires set_page_config to be the first Streamlit command run.
st.set_page_config(
    page_title="Multimodal AI Assistant",
    page_icon="🤖",
    layout="centered",
)


# Rendering a page is deterministic and relatively expensive. Streamlit keys
# this cache by the PDF bytes and page number, so revisiting a page is fast.
# The cache is cleared when a new document is loaded.
@st.cache_data(show_spinner=False, max_entries=40)
def _cached_render_pdf_page(pdf_bytes: bytes, page_number: int) -> bytes:
    """Return cached PNG bytes for one rendered PDF page.

    Streamlit hashes the arguments, so the same document and page number reuse
    the stored image instead of rendering the page again.
    """
    return render_pdf_page(pdf_bytes, page_number)


def _reset_image_state() -> None:
    """Forget the current image's answer, errors, and question history."""
    st.session_state.image_signature = None
    st.session_state.image_info = None
    st.session_state.image_bytes = None
    st.session_state.image_error = None
    st.session_state.image_answer = None
    st.session_state.image_history = []


def _init_image_state() -> None:
    """Create the session keys used by the image question answering page."""
    # session_state keeps values for one browser session across Streamlit reruns.
    if "image_uploader_key" not in st.session_state:
        # Changing this key gives us a fresh file_uploader for the Clear button.
        st.session_state.image_uploader_key = 0
    if "image_signature" not in st.session_state:
        _reset_image_state()


def _render_image_qa() -> None:
    """Render upload, preview, question form, answer, and history."""
    # The page order is: upload -> validate/preview -> ask -> show answer/history.

    # `type` is only a browser-side filter. The real content check happens in
    # validate_image(), so renaming a .txt file to .png is still rejected.
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=IMAGE_FILE_TYPES,
        key=f"image_uploader_{st.session_state.image_uploader_key}",
        help=(
            "JPEG, PNG, or WEBP. "
            f"Maximum size: {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
        ),
    )

    if uploaded_file is None:
        # Removing the file should not leave an old answer on screen.
        if st.session_state.image_signature is not None:
            _reset_image_state()
        st.info("Upload a JPEG, PNG, or WEBP image to get started.")
        return

    image_bytes = uploaded_file.getvalue()

    # A hash identifies the current image without storing the whole file twice.
    # If the hash changes, the old answer and history belong to another image.
    signature = hashlib.sha256(image_bytes).hexdigest()
    if signature != st.session_state.image_signature:
        _reset_image_state()
        st.session_state.image_signature = signature

        # Streamlit reruns this script often, so validate and cache the metadata
        # once per image instead of reopening the file for every interaction.
        try:
            validate_image(image_bytes)
            st.session_state.image_info = get_image_info(image_bytes)
            # Keep the validated bytes in memory so the voice assistant can
            # reuse the current image without uploading it again.
            st.session_state.image_bytes = image_bytes
        except ImageValidationError as error:
            st.session_state.image_bytes = None
            st.session_state.image_error = str(error)

    if st.session_state.image_error:
        st.error(st.session_state.image_error)
    else:
        # Preview the original upload. The AI receives a resized copy later.
        st.image(image_bytes, caption=uploaded_file.name)

        info = st.session_state.image_info
        if info:
            size_mb = int(info["size_bytes"]) / (1024 * 1024)
            st.caption(
                f"{info['format']} · {info['width']} × {info['height']} pixels "
                f"· {size_mb:.2f} MB"
            )

    if st.button("Clear image and answers"):
        # A new uploader key empties the widget on the next rerun.
        st.session_state.image_uploader_key += 1
        _reset_image_state()
        st.rerun()

    if st.session_state.image_error:
        return

    st.divider()
    st.subheader("Ask about this image")

    # A form prevents an API call on every widget rerun. The call happens only
    # after the user presses Ask AI. The key changes with the image, so a new
    # upload starts with a fresh, empty question box.
    with st.form(key=f"image_question_form_{signature[:12]}"):
        question = st.text_input(
            "Your question",
            placeholder="What is in this image?",
            max_chars=MAX_QUESTION_LENGTH,
        )
        submitted = st.form_submit_button("Ask AI", type="primary")

    # Everything below this point runs only after the Ask AI button is clicked.
    if submitted:
        if not question.strip():
            st.warning("Please enter a question about the image.")
        else:
            try:
                # Prepare a smaller RGB copy so we do not upload the original
                # large file or send an unsupported color mode to the API.
                with st.spinner("Analyzing the image..."):
                    prepared_bytes, prepared_mime = prepare_image(image_bytes)
                    answer = ask_image(question, prepared_bytes, prepared_mime)
            except ConfigurationError as error:
                st.error(f"AI is not configured: {error}")
            except (ImageValidationError, AIAssistantError) as error:
                st.error(str(error))
            else:
                # The latest answer gets its own section; the list keeps all
                # question/answer pairs for the current uploaded image only.
                st.session_state.image_answer = answer
                st.session_state.image_history.append(
                    {"question": question.strip(), "answer": answer}
                )

    # These values survive reruns, so the answer stays visible after the user
    # changes a widget or submits another question.
    if st.session_state.image_answer:
        st.subheader("Latest answer")
        st.markdown(st.session_state.image_answer)

    if st.session_state.image_history:
        with st.expander(
            f"Question history for this image "
            f"({len(st.session_state.image_history)})"
        ):
            for item in reversed(st.session_state.image_history):
                st.markdown(f"**Question:** {item['question']}")
                st.markdown(item["answer"])
                st.divider()


def _reset_pdf_state() -> None:
    """Forget the current PDF's text, answers, and question history."""
    st.session_state.pdf_signature = None
    st.session_state.pdf_info = None
    st.session_state.pdf_bytes = None
    st.session_state.pdf_pages = []
    st.session_state.pdf_error = None
    st.session_state.pdf_answer = None
    st.session_state.pdf_answer_pages = []
    st.session_state.pdf_history = []
    st.session_state.pdf_page_number = 1


def _init_pdf_state() -> None:
    """Create the session keys used by the PDF question answering page."""
    if "pdf_uploader_key" not in st.session_state:
        # Incrementing this key gives the Clear button a fresh file uploader.
        st.session_state.pdf_uploader_key = 0
    if "pdf_signature" not in st.session_state:
        _reset_pdf_state()


def _build_pdf_page_context(
    pdf_bytes: bytes,
    pages: list[dict[str, object]],
    selected_pages: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Combine selected page text with fresh rendered PNG images."""
    # Look pages up by their 1-based number so the text and image always belong
    # to the same page, even if the selection order changes.
    page_lookup = {int(page["page_number"]): page for page in pages}
    context = []

    for selected_page in selected_pages:
        page_number = int(selected_page["page_number"])
        page = page_lookup[page_number]
        context.append(
            {
                "page_number": page_number,
                "text": str(page["text"]),
                # Rendering is cached, so asking two questions about the same
                # page does not render the PDF page twice.
                "image_bytes": _cached_render_pdf_page(pdf_bytes, page_number),
                "mime_type": "image/png",
            }
        )

    return context


def _render_pdf_qa() -> None:
    """Render PDF upload, page preview, question modes, answers, and history."""
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        key=f"pdf_uploader_{st.session_state.pdf_uploader_key}",
        help=(
            "Maximum size: "
            f"{MAX_PDF_BYTES // (1024 * 1024)} MB. "
            f"Maximum length: {MAX_PDF_PAGES} pages."
        ),
    )

    if uploaded_file is None:
        if st.session_state.pdf_signature is not None:
            _reset_pdf_state()
        st.info("Upload a PDF to get started.")
        return

    pdf_bytes = uploaded_file.getvalue()

    # The content hash identifies the document. A different hash means every
    # old answer, page selection, and extracted page belongs to another file.
    signature = hashlib.sha256(pdf_bytes).hexdigest()
    if signature != st.session_state.pdf_signature:
        _reset_pdf_state()
        st.session_state.pdf_signature = signature
        _cached_render_pdf_page.clear()

        try:
            validate_pdf(pdf_bytes)
            st.session_state.pdf_info = get_pdf_info(pdf_bytes)
            st.session_state.pdf_pages = extract_pdf_text(pdf_bytes)
            # Keep the validated PDF in memory so the voice assistant can ask
            # about the current document without a second upload.
            st.session_state.pdf_bytes = pdf_bytes
        except PDFProcessingError as error:
            st.session_state.pdf_bytes = None
            st.session_state.pdf_error = str(error)

    st.markdown(f"**{uploaded_file.name}**")

    info = st.session_state.pdf_info
    if info:
        size_mb = int(info["size_bytes"]) / (1024 * 1024)
        st.caption(
            f"{info['page_count']} pages · {size_mb:.2f} MB · "
            f"rendered at {RENDER_DPI} DPI"
        )

    if st.button("Clear document and answers"):
        st.session_state.pdf_uploader_key += 1
        _reset_pdf_state()
        _cached_render_pdf_page.clear()
        st.rerun()

    if st.session_state.pdf_error:
        st.error(st.session_state.pdf_error)
        return

    page_count = int(st.session_state.pdf_info["page_count"])
    pages = st.session_state.pdf_pages

    st.divider()
    page_number = st.selectbox(
        "Preview page",
        options=list(range(1, page_count + 1)),
        key=f"pdf_page_selector_{signature[:12]}",
    )
    # Remember the current selection in session state so it can be reused by
    # other parts of the page (for example, a future follow-up question).
    st.session_state.pdf_page_number = page_number

    # The preview always shows the full rendered page, not only extracted text.
    try:
        with st.spinner(f"Rendering page {page_number}..."):
            page_png = _cached_render_pdf_page(pdf_bytes, page_number)
        st.image(page_png, caption=f"Page {page_number} of {page_count}")
    except PDFProcessingError as error:
        st.error(str(error))
        return

    st.divider()
    question_mode = st.radio(
        "Question mode",
        options=["Selected page", "Whole document"],
        horizontal=True,
        key=f"pdf_question_mode_{signature[:12]}",
    )

    if question_mode == "Selected page":
        st.caption(
            "The selected page's text and rendered image are both sent to the AI."
        )
    else:
        st.caption(
            "Up to 3 pages are chosen with simple keyword matching. "
            "This is a heuristic, not semantic search."
        )

    # A form means the API is called only after Ask AI is pressed.
    with st.form(key=f"pdf_question_form_{signature[:12]}_{question_mode}"):
        question = st.text_input(
            "Your question",
            placeholder="Summarize page 2 or explain the diagram on page 5.",
            max_chars=MAX_QUESTION_LENGTH,
        )
        submitted = st.form_submit_button("Ask AI", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Please enter a question about the PDF.")
        else:
            try:
                # Selected Page mode sends exactly one page. Whole Document mode
                # uses keyword scoring to choose a small set of relevant pages.
                if question_mode == "Selected page":
                    selected_pages = [pages[page_number - 1]]
                else:
                    selected_pages = select_relevant_pages(pages, question)

                with st.spinner("Reading the selected PDF pages..."):
                    page_context = _build_pdf_page_context(
                        pdf_bytes,
                        pages,
                        selected_pages,
                    )
                    answer = ask_pdf(question, page_context)
            except PDFProcessingError as error:
                st.error(str(error))
            except ConfigurationError as error:
                st.error(f"AI is not configured: {error}")
            except AIAssistantError as error:
                st.error(str(error))
            else:
                pages_used = [int(page["page_number"]) for page in selected_pages]
                st.session_state.pdf_answer = answer
                st.session_state.pdf_answer_pages = pages_used
                st.session_state.pdf_history.append(
                    {
                        "question": question.strip(),
                        "answer": answer,
                        "mode": question_mode,
                        "pages": pages_used,
                    }
                )

    if st.session_state.pdf_answer:
        st.subheader("Latest answer")
        st.markdown(st.session_state.pdf_answer)
        if st.session_state.pdf_answer_pages:
            page_labels = ", ".join(
                str(page) for page in st.session_state.pdf_answer_pages
            )
            st.caption(f"Pages analyzed: {page_labels}")

    if st.session_state.pdf_history:
        with st.expander(
            f"Question history for this PDF "
            f"({len(st.session_state.pdf_history)})"
        ):
            for item in reversed(st.session_state.pdf_history):
                page_labels = ", ".join(str(page) for page in item["pages"])
                st.markdown(
                    f"**Question:** {item['question']}  \n"
                    f"**Mode:** {item['mode']} · **Pages:** {page_labels}"
                )
                st.markdown(item["answer"])
                st.divider()


def _clear_voice_results() -> None:
    """Clear the transcription result, answer, and spoken audio."""
    st.session_state.voice_transcript = ""
    st.session_state.voice_error = None
    st.session_state.voice_answer = None
    st.session_state.voice_answer_pages = []
    st.session_state.voice_answer_question = None
    st.session_state.voice_answer_scope = None
    st.session_state.voice_tts_audio = None
    st.session_state.voice_tts_answer_hash = None
    st.session_state.voice_tts_error = None


def _reset_voice_recording() -> None:
    """Forget the current recording while keeping the chosen context."""
    st.session_state.voice_audio_signature = None
    st.session_state.voice_audio_bytes = None
    _clear_voice_results()


def _reset_voice_interaction() -> None:
    """Clear the recording and every result for the voice interaction."""
    # A new widget key gives the user an empty recorder on the next rerun.
    st.session_state.voice_audio_key += 1
    _reset_voice_recording()


def _init_voice_state() -> None:
    """Create the session keys used by the voice assistant page."""
    if "voice_audio_key" not in st.session_state:
        st.session_state.voice_audio_key = 0
    if "voice_context" not in st.session_state:
        st.session_state.voice_context = "General"
    if "voice_pdf_scope" not in st.session_state:
        st.session_state.voice_pdf_scope = "Selected page"
    if "voice_audio_signature" not in st.session_state:
        _reset_voice_recording()


def _invalidate_voice_answer() -> None:
    """Drop reusable results when the question or context changes.

    This is used as a widget callback, so an old answer or spoken file cannot
    be shown for a newly edited question or a different context.
    """
    st.session_state.voice_answer = None
    st.session_state.voice_answer_pages = []
    st.session_state.voice_answer_question = None
    st.session_state.voice_answer_scope = None
    st.session_state.voice_tts_audio = None
    st.session_state.voice_tts_answer_hash = None
    st.session_state.voice_tts_error = None


def _answer_voice_question(
    question: str,
    context: str,
    pdf_scope: str | None,
    pdf_page: int | None,
) -> tuple[str, list[int], str]:
    """Route a transcribed question to the existing text, image, or PDF path.

    Routing is explicit: the user chooses the context, so no intent detection
    or autonomous agent is involved.
    """
    if context == "General":
        return ask_text(question), [], "general"

    if context == "Current Image":
        # The image bytes were kept in session state when the image was
        # uploaded, so the voice page can reuse the same image pipeline.
        image_bytes = st.session_state.image_bytes
        prepared_bytes, prepared_mime = prepare_image(image_bytes)
        answer = ask_image(question, prepared_bytes, prepared_mime)
        scope = f"image:{st.session_state.image_signature}"
        return answer, [], scope

    # PDF context: reuse the stored pages and the selected page or the same
    # keyword-based whole-document selection as the PDF page.
    pdf_bytes = st.session_state.pdf_bytes
    pages = st.session_state.pdf_pages

    if pdf_scope == "Selected page":
        if pdf_page is None:
            raise VoiceProcessingError("Choose a PDF page before asking.")
        page_number = int(pdf_page)
        selected_pages = [pages[page_number - 1]]
        scope = f"pdf:{st.session_state.pdf_signature}:{page_number}"
    else:
        selected_pages = select_relevant_pages(pages, question)
        scope = f"pdf:{st.session_state.pdf_signature}:all"

    page_context = _build_pdf_page_context(pdf_bytes, pages, selected_pages)
    answer = ask_pdf(question, page_context)
    page_numbers = [int(page["page_number"]) for page in selected_pages]
    return answer, page_numbers, scope


def _render_voice_qa() -> None:
    """Render recording, transcription, routing, answer, and speech output."""
    has_image = st.session_state.image_bytes is not None
    has_pdf = st.session_state.pdf_bytes is not None

    context = st.radio(
        "Question context",
        options=list(VOICE_CONTEXTS),
        horizontal=True,
        key="voice_context",
        on_change=_invalidate_voice_answer,
        help=(
            "General asks a normal question. Current Image and Current PDF "
            "reuse a file already uploaded in those modes."
        ),
    )

    # Show clearly whether the selected context has a file to work with.
    if context == "Current Image":
        if has_image:
            info = st.session_state.image_info or {}
            st.caption(
                f"Using the current image: {info.get('width')} × "
                f"{info.get('height')} pixels."
            )
        else:
            st.warning(
                "No image is loaded. Upload one in Image Question Answering "
                "mode first, or choose the General context."
            )
    elif context == "Current PDF":
        if has_pdf:
            info = st.session_state.pdf_info or {}
            st.caption(
                f"Using the current PDF: {info.get('page_count')} pages."
            )
        else:
            st.warning(
                "No PDF is loaded. Upload one in PDF Question Answering mode "
                "first, or choose the General context."
            )

    # PDF questions need to know which page or how much of the document to use.
    pdf_scope = None
    pdf_page = None
    if context == "Current PDF" and has_pdf:
        pdf_scope = st.radio(
            "PDF question scope",
            options=["Selected page", "Whole document"],
            horizontal=True,
            key="voice_pdf_scope",
            on_change=_invalidate_voice_answer,
        )
        if pdf_scope == "Selected page":
            page_count = int(st.session_state.pdf_info["page_count"])
            default_page = min(
                max(1, int(st.session_state.pdf_page_number)),
                page_count,
            )
            pdf_page = st.selectbox(
                "PDF page",
                options=list(range(1, page_count + 1)),
                index=default_page - 1,
                key="voice_pdf_page",
                on_change=_invalidate_voice_answer,
            )

    if st.button("Reset interaction"):
        _reset_voice_interaction()
        st.rerun()

    st.divider()

    # The browser records the audio and sends WAV bytes to Streamlit. No
    # server-side microphone or PyAudio object is used.
    audio_file = st.audio_input(
        f"Record a question (up to {MAX_AUDIO_SECONDS} seconds)",
        key=f"voice_audio_{st.session_state.voice_audio_key}",
        sample_rate=16000,
        help="Press the microphone to start and stop recording.",
    )

    if audio_file is None:
        if st.session_state.voice_audio_signature is not None:
            _reset_voice_recording()
        st.info(
            "Press the microphone, ask your question, and stop the recording. "
            "The transcription will appear here."
        )
        return

    audio_bytes = audio_file.getvalue()

    # Validate and identify the recording. The hash lets the app skip
    # transcription when Streamlit reruns with the same audio.
    try:
        signature = audio_signature(audio_bytes)
    except VoiceProcessingError as error:
        _reset_voice_recording()
        st.session_state.voice_error = str(error)
        signature = None

    if signature and signature != st.session_state.voice_audio_signature:
        # A new recording invalidates the old question, answer, and speech.
        _reset_voice_recording()
        st.session_state.voice_audio_signature = signature
        st.session_state.voice_audio_bytes = audio_bytes

        try:
            with st.spinner("Transcribing your question..."):
                _signature, transcript, _transcribed = transcribe_audio(audio_bytes)
        except VoiceProcessingError as error:
            st.session_state.voice_error = str(error)
        else:
            st.session_state.voice_transcript = transcript
            # Seed the editable text area for this specific recording.
            editor_key = f"voice_editor_{signature[:12]}"
            st.session_state[editor_key] = transcript

    # Always preview the recorded audio once it has been captured.
    st.audio(audio_bytes, format="audio/wav")
    try:
        info = get_audio_info(audio_bytes)
    except VoiceProcessingError:
        info = None
    if info:
        st.caption(
            f"Recording: {info['duration_seconds']:.1f} s · "
            f"{info['sample_rate']} Hz · {info['channels']} channel(s)"
        )

    if st.session_state.voice_error:
        st.error(st.session_state.voice_error)
        return

    if not st.session_state.voice_transcript:
        return

    editor_key = f"voice_editor_{st.session_state.voice_audio_signature[:12]}"
    edited_question = st.text_area(
        "Transcription (edit it if speech recognition made a mistake)",
        key=editor_key,
        max_chars=MAX_QUESTION_LENGTH,
        height=100,
        on_change=_invalidate_voice_answer,
    )
    st.caption(
        "Speech recognition uses an online service, so it needs internet access "
        "and may not be perfectly accurate."
    )

    if st.button("Submit Question", type="primary"):
        question = (edited_question or "").strip()
        if not question:
            st.warning("Please record or type a question first.")
        else:
            try:
                resolve_voice_context(context, has_image, has_pdf)
                if context == "Current PDF" and pdf_scope is None:
                    raise VoiceProcessingError(
                        "Choose a PDF question scope before asking."
                    )

                with st.spinner("Asking the AI..."):
                    answer, pages_used, scope_used = _answer_voice_question(
                        question,
                        context,
                        pdf_scope,
                        pdf_page,
                    )
            except ConfigurationError as error:
                st.error(f"AI is not configured: {error}")
            except (
                VoiceProcessingError,
                AIAssistantError,
                ImageValidationError,
                PDFProcessingError,
            ) as error:
                st.error(str(error))
            else:
                st.session_state.voice_answer = answer
                st.session_state.voice_answer_pages = pages_used
                st.session_state.voice_answer_question = question
                st.session_state.voice_answer_scope = scope_used
                # A new answer makes any previously generated speech stale.
                st.session_state.voice_tts_audio = None
                st.session_state.voice_tts_answer_hash = None
                st.session_state.voice_tts_error = None

    # Work out which context the answer would belong to right now. An answer
    # is shown only when it matches, so an old image/PDF answer cannot appear.
    if context == "General":
        current_scope = "general"
    elif context == "Current Image":
        current_scope = f"image:{st.session_state.image_signature}"
    elif context == "Current PDF" and has_pdf:
        if pdf_scope == "Selected page":
            current_scope = f"pdf:{st.session_state.pdf_signature}:{pdf_page}"
        else:
            current_scope = f"pdf:{st.session_state.pdf_signature}:all"
    else:
        current_scope = None

    if st.session_state.voice_answer:
        if current_scope and st.session_state.voice_answer_scope == current_scope:
            st.subheader("AI answer")
            st.markdown(st.session_state.voice_answer)
            if st.session_state.voice_answer_pages:
                page_labels = ", ".join(
                    str(page) for page in st.session_state.voice_answer_pages
                )
                st.caption(f"Pages analyzed: {page_labels}")

            # Speak the answer only on request. The generated WAV bytes are
            # cached by answer text, so reruns do not synthesize again.
            answer_hash = hashlib.sha256(
                st.session_state.voice_answer.encode("utf-8")
            ).hexdigest()

            if st.button("Generate Speech"):
                if not (
                    st.session_state.voice_tts_audio
                    and st.session_state.voice_tts_answer_hash == answer_hash
                ):
                    try:
                        with st.spinner("Generating speech..."):
                            speech_bytes = text_to_speech(
                                st.session_state.voice_answer
                            )
                    except VoiceProcessingError as error:
                        st.session_state.voice_tts_error = str(error)
                        st.session_state.voice_tts_audio = None
                        st.session_state.voice_tts_answer_hash = None
                    else:
                        st.session_state.voice_tts_audio = speech_bytes
                        st.session_state.voice_tts_answer_hash = answer_hash
                        st.session_state.voice_tts_error = None

            if (
                st.session_state.voice_tts_audio
                and st.session_state.voice_tts_answer_hash == answer_hash
            ):
                st.audio(st.session_state.voice_tts_audio, format="audio/wav")
            elif st.session_state.voice_tts_error:
                st.warning(st.session_state.voice_tts_error)
        else:
            st.info(
                "The question context changed. Submit the question again to get "
                "an answer for the current selection."
            )


st.title("Multimodal AI Assistant")
st.write(
    "A college mini project that answers questions about images, PDFs, and "
    "spoken questions, including optional spoken answers."
)
st.caption("Phase 4: voice assistant integration")

_init_image_state()
_init_pdf_state()
_init_voice_state()

# Streamlit reruns the whole script when a widget changes, so the selected
# mode is read directly and used to render the rest of the page.
with st.sidebar:
    st.header("Modes")
    selected_mode = st.sidebar.radio("Choose a mode", list(MODES))
    st.divider()

    st.subheader("Configuration")
    # Validate configuration for display only. No client or API call is created
    # here, so the app can open without a Groq key.
    try:
        settings = get_settings()
    except ConfigurationError as error:
        st.warning("AI features are not configured yet.")
        st.caption(str(error))
    else:
        st.success("Groq API key loaded.")
        st.caption(f"Text model: {settings.model}")
        st.caption(f"Vision model: {settings.vision_model}")
    st.caption("The API key is read from your local .env file and is never displayed.")

st.subheader(f"Selected mode: {selected_mode}")
st.info(MODES[selected_mode])
st.divider()

if selected_mode == "Image Question Answering":
    _render_image_qa()
elif selected_mode == "PDF Question Answering":
    _render_pdf_qa()
else:
    _render_voice_qa()

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Images: available**")
    st.caption("Upload an image and ask questions about it.")
with col2:
    st.markdown("**PDFs: available**")
    st.caption("Ask about text, diagrams, tables, and scanned pages.")
with col3:
    st.markdown("**Voice: available**")
    st.caption("Record a question and optionally hear the answer.")

st.caption("Multimodal AI Assistant · Phase 4")
