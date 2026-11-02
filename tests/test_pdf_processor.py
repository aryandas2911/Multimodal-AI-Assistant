"""Tests for PDF validation, extraction, rendering, and PDF Q&A."""

from io import BytesIO
from types import SimpleNamespace

import httpx
import pymupdf
import pytest
from groq import RateLimitError
from PIL import Image

from modules.ai_assistant import (
    PDF_SYSTEM_INSTRUCTION,
    AIAssistantError,
    AppSettings,
    ask_pdf,
)
from modules.pdf_processor import (
    MAX_PDF_BYTES,
    MAX_PDF_PAGES,
    PDFProcessingError,
    extract_pdf_text,
    get_pdf_info,
    render_pdf_page,
    select_relevant_pages,
    validate_pdf,
)

TEST_SETTINGS = AppSettings(
    api_key="gsk_test_key",
    model="test-text-model",
    vision_model="test-vision-model",
)


def make_text_pdf(page_texts: list[str]) -> bytes:
    """Create a small text PDF entirely in memory."""
    document = pymupdf.open()
    try:
        for text in page_texts:
            page = document.new_page()
            if text:
                page.insert_textbox(
                    pymupdf.Rect(72, 72, 523, 770),
                    text,
                    fontsize=11,
                )
        return document.tobytes()
    finally:
        document.close()


def make_image_only_pdf(page_count: int = 1) -> bytes:
    """Create a PDF whose pages contain an image but no selectable text."""
    buffer = BytesIO()
    Image.new("RGB", (200, 120), "orange").save(buffer, format="PNG")
    png_bytes = buffer.getvalue()

    document = pymupdf.open()
    try:
        for _ in range(page_count):
            page = document.new_page()
            page.insert_image(pymupdf.Rect(72, 72, 272, 192), stream=png_bytes)
        return document.tobytes()
    finally:
        document.close()


class FakeCompletions:
    """Capture the PDF request without making a real API call."""

    def __init__(self, content="The answer is on page 2.", error=None):
        self.content = content
        self.error = error
        self.last_call = None

    def create(self, **kwargs):
        self.last_call = kwargs
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )


class FakeClient:
    """Minimal client shaped like the Groq SDK client used by ask_pdf()."""

    def __init__(self, content="The answer is on page 2.", error=None):
        self.completions = FakeCompletions(content=content, error=error)
        self.chat = SimpleNamespace(completions=self.completions)


def test_valid_pdf_is_accepted():
    pdf_bytes = make_text_pdf(["A valid one-page PDF."])

    validate_pdf(pdf_bytes)
    info = get_pdf_info(pdf_bytes)

    assert info["page_count"] == 1
    assert info["mime_type"] == "application/pdf"
    assert info["size_bytes"] == len(pdf_bytes)


def test_empty_pdf_bytes_are_rejected():
    with pytest.raises(PDFProcessingError, match="empty"):
        validate_pdf(b"")


def test_invalid_pdf_bytes_are_rejected():
    with pytest.raises(PDFProcessingError, match="not a valid PDF"):
        validate_pdf(b"this is not a PDF")


def test_multipage_text_has_correct_page_numbers():
    pdf_bytes = make_text_pdf(
        [
            "Introduction to operating systems.",
            "Paging divides virtual memory into fixed-size pages.",
            "Segmentation divides memory into logical segments.",
        ]
    )

    pages = extract_pdf_text(pdf_bytes)

    assert len(pages) == 3
    assert [page["page_number"] for page in pages] == [1, 2, 3]
    assert "Paging" in pages[1]["text"]
    assert "Segmentation" in pages[2]["text"]


def test_text_extraction_normalizes_whitespace():
    pdf_bytes = make_text_pdf(["Paging    divides   memory into pages."])

    pages = extract_pdf_text(pdf_bytes)

    assert "Paging divides memory into pages." in pages[0]["text"]
    assert "  " not in pages[0]["text"]


def test_image_only_page_has_no_extracted_text():
    pdf_bytes = make_image_only_pdf()

    pages = extract_pdf_text(pdf_bytes)

    assert pages[0]["has_text"] is False
    assert pages[0]["text"] == ""


def test_page_rendering_returns_png():
    pdf_bytes = make_text_pdf(["Page with a diagram placeholder."])

    png_bytes = render_pdf_page(pdf_bytes, 1)

    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(BytesIO(png_bytes)) as image:
        assert image.width > 0
        assert image.height > 0


def test_image_only_page_can_still_be_rendered():
    pdf_bytes = make_image_only_pdf()

    png_bytes = render_pdf_page(pdf_bytes, 1)

    assert len(png_bytes) > 0
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("page_number", [0, 4, True])
def test_invalid_page_selection_is_rejected(page_number):
    pdf_bytes = make_text_pdf(["Page 1", "Page 2", "Page 3"])

    with pytest.raises(PDFProcessingError):
        render_pdf_page(pdf_bytes, page_number)


def test_encrypted_pdf_is_rejected():
    document = pymupdf.open()
    try:
        page = document.new_page()
        page.insert_text((72, 72), "Secret content")
        encrypted_bytes = document.tobytes(
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            owner_pw="owner-password",
            user_pw="user-password",
        )
    finally:
        document.close()

    with pytest.raises(PDFProcessingError, match="encrypted"):
        validate_pdf(encrypted_bytes)


def test_oversized_pdf_is_rejected():
    oversized_bytes = b"%PDF-1.4" + b"\x00" * (MAX_PDF_BYTES + 1)

    with pytest.raises(PDFProcessingError, match="too large"):
        validate_pdf(oversized_bytes)


def test_page_count_limit_is_enforced():
    pdf_bytes = make_text_pdf(["Page"] * (MAX_PDF_PAGES + 1))

    with pytest.raises(PDFProcessingError, match="Maximum supported length"):
        validate_pdf(pdf_bytes)


def test_keyword_matching_selects_relevant_page():
    pages = extract_pdf_text(
        make_text_pdf(
            [
                "Introduction to computer architecture.",
                "Paging uses fixed-size pages and a page table.",
                "Segmentation uses variable-size logical segments.",
            ]
        )
    )

    selected = select_relevant_pages(pages, "How does segmentation work?")

    assert [page["page_number"] for page in selected] == [3]


def test_explicit_page_reference_is_respected():
    pages = extract_pdf_text(make_text_pdf(["Page one", "Page two", "Page three"]))

    selected = select_relevant_pages(pages, "Summarize page 3")

    assert [page["page_number"] for page in selected] == [3]


def test_summary_question_uses_first_available_pages():
    pages = extract_pdf_text(
        make_text_pdf(["Page one", "Page two", "Page three", "Page four"])
    )

    selected = select_relevant_pages(pages, "List the important topics.")

    assert [page["page_number"] for page in selected] == [1, 2, 3]


def test_too_many_explicit_pages_are_rejected():
    pages = extract_pdf_text(
        make_text_pdf(["Page one", "Page two", "Page three", "Page four"])
    )

    with pytest.raises(PDFProcessingError, match="at most 3 pages"):
        select_relevant_pages(
            pages,
            "Compare page 1, page 2, page 3, and page 4.",
        )


def test_low_confidence_selection_asks_for_manual_page():
    pages = extract_pdf_text(
        make_text_pdf(
            [
                "Alpha content.",
                "Beta content.",
                "Gamma content.",
                "Delta content.",
            ]
        )
    )

    with pytest.raises(PDFProcessingError, match="manually"):
        select_relevant_pages(pages, "Explain quantum entanglement.")


def test_ask_pdf_includes_text_and_page_images():
    fake_client = FakeClient(content="Paging is explained on page 2.")
    page_context = [
        {
            "page_number": 2,
            "text": "Paging divides memory into fixed-size pages.",
            "image_bytes": b"page-2-png",
            "mime_type": "image/png",
        },
        {
            "page_number": 3,
            "text": "",  # Simulates a scanned page with no selectable text.
            "image_bytes": b"page-3-png",
            "mime_type": "image/png",
        },
    ]

    answer = ask_pdf(
        "Explain paging and the diagram.",
        page_context,
        client=fake_client,
        settings=TEST_SETTINGS,
    )

    assert answer == "Paging is explained on page 2."
    call = fake_client.completions.last_call
    assert call["model"] == "test-vision-model"
    assert call["messages"][0] == {
        "role": "system",
        "content": PDF_SYSTEM_INSTRUCTION,
    }

    content_parts = call["messages"][1]["content"]
    prompt = content_parts[0]["text"]
    assert "PAGE 2 TEXT" in prompt
    assert "PAGE 3 TEXT" in prompt
    assert "No selectable text" in prompt

    image_parts = [part for part in content_parts if part["type"] == "image_url"]
    assert len(image_parts) == 2
    assert image_parts[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert image_parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_ask_pdf_requires_page_context():
    with pytest.raises(AIAssistantError, match="No PDF page context"):
        ask_pdf(
            "What is on this page?",
            [],
            client=FakeClient(),
            settings=TEST_SETTINGS,
        )


def test_ask_pdf_handles_api_errors_safely():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request)
    error = RateLimitError("raw provider details", response=response, body=None)
    page_context = [
        {
            "page_number": 1,
            "text": "Some text",
            "image_bytes": b"page-png",
            "mime_type": "image/png",
        }
    ]

    with pytest.raises(AIAssistantError, match="rate limit or quota"):
        ask_pdf(
            "Summarize this page.",
            page_context,
            client=FakeClient(error=error),
            settings=TEST_SETTINGS,
        )
