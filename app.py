"""Streamlit entry point for the Multimodal AI Assistant (Phase 2)."""

import hashlib

import streamlit as st

from modules.ai_assistant import (
    AIAssistantError,
    ConfigurationError,
    MAX_QUESTION_LENGTH,
    ask_image,
    get_settings,
)
from modules.image_processor import (
    MAX_IMAGE_BYTES,
    ImageValidationError,
    get_image_info,
    prepare_image,
    validate_image,
)

# Modes are defined once and reused by both the sidebar and the page content.
MODES = {
    "Image Question Answering": "Ask questions about an uploaded image.",
    "PDF Question Answering": (
        "Ask questions about PDF text, images, tables, and diagrams. The PDF "
        "pipeline will be added in a later phase."
    ),
    "Voice Assistant": (
        "Speak a question and hear the answer. The voice pipeline will be "
        "added in a later phase."
    ),
}

IMAGE_FILE_TYPES = ["jpg", "jpeg", "png", "webp"]

# Streamlit requires set_page_config to be the first Streamlit command run.
st.set_page_config(
    page_title="Multimodal AI Assistant",
    page_icon="🤖",
    layout="centered",
)


def _reset_image_state() -> None:
    """Forget the current image's answer, errors, and question history."""
    st.session_state.image_signature = None
    st.session_state.image_info = None
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
        except ImageValidationError as error:
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


st.title("Multimodal AI Assistant")
st.write(
    "A college mini project that answers questions about images and will later "
    "support PDFs and voice-based conversations with spoken responses."
)
st.caption("Phase 2: image question answering")

_init_image_state()

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
else:
    with st.container(border=True):
        st.markdown("### Feature coming in a later phase")
        st.write(
            f"**{selected_mode}** is not implemented yet. Phase 2 implements "
            "image question answering only."
        )
        st.write(
            "PDF processing and voice interaction will be connected to their "
            "modules in later phases."
        )

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Images: available**")
    st.caption("Upload an image and ask questions about it.")
with col2:
    st.markdown("**PDFs: planned**")
    st.caption("Document question answering across text and visuals.")
with col3:
    st.markdown("**Voice: planned**")
    st.caption("Speech input with optional spoken replies.")

st.caption("Multimodal AI Assistant · Phase 2")
