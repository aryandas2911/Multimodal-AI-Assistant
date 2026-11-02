"""Streamlit entry point for the Multimodal AI Assistant (Phase 1)."""

import streamlit as st

from modules.ai_assistant import ConfigurationError, get_settings

# Modes are defined once and reused by both the sidebar and the page content.
MODES = {
    "Image Question Answering": (
        "Ask questions about an uploaded image. The image pipeline will be "
        "added in a later phase."
    ),
    "PDF Question Answering": (
        "Ask questions about PDF text, images, tables, and diagrams. The PDF "
        "pipeline will be added in a later phase."
    ),
    "Voice Assistant": (
        "Speak a question and hear the answer. The voice pipeline will be "
        "added in a later phase."
    ),
}

# Streamlit requires set_page_config to be the first Streamlit command run.
st.set_page_config(
    page_title="Multimodal AI Assistant",
    page_icon="🤖",
    layout="centered",
)

st.title("Multimodal AI Assistant")
st.write(
    "A college mini project that will answer questions about images and PDFs, "
    "and support voice-based conversations with spoken responses."
)
st.caption("Phase 1: project setup and application shell")

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
        st.caption(f"Model: {settings.model}")
    st.caption("The API key is read from your local .env file and is never displayed.")

st.subheader(f"Selected mode: {selected_mode}")
st.info(MODES[selected_mode])

st.divider()

with st.container(border=True):
    st.markdown("### Feature coming in a later phase")
    st.write(
        f"**{selected_mode}** is not implemented yet. Phase 1 only provides "
        "the project structure, configuration, and this runnable interface."
    )
    st.write(
        "The navigation, session state, file upload, chat, and audio controls "
        "will be connected to the processing modules in the next phases."
    )

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Planned: Images**")
    st.caption("Image understanding and visual question answering.")
with col2:
    st.markdown("**Planned: PDFs**")
    st.caption("Document question answering across text and visuals.")
with col3:
    st.markdown("**Planned: Voice**")
    st.caption("Speech input with optional spoken replies.")

st.caption("Multimodal AI Assistant · Phase 1")
