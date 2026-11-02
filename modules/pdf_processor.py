"""PDF validation, text extraction, and page rendering.

PyMuPDF is used for two things:
1. Reading selectable text (which can be empty on scanned pages).
2. Rendering the whole page as a PNG so diagrams, tables, and scanned content
   are still visible to the multimodal AI model.
"""

import re

import pymupdf

MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 30
RENDER_DPI = 144

# A very large page at 144 DPI can use a lot of memory. This cap keeps the
# mini-project version predictable while still allowing normal A4/Letter pages.
MAX_RENDER_PIXELS = 12_000_000

# Sending one image per page is expensive, so a whole-document question uses
# at most three rendered pages. This also matches the vision model's limits.
MAX_RELEVANT_PAGES = 3

SUMMARY_WORDS = {
    "summary",
    "summarize",
    "overview",
    "topics",
    "list",
    "main",
    "points",
}

# Common words are ignored during keyword matching because they appear on
# almost every page and would not help us choose relevant pages.
STOPWORDS = {
    "a",
    "about",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "between",
    "but",
    "by",
    "can",
    "could",
    "describe",
    "did",
    "difference",
    "discussed",
    "do",
    "does",
    "explain",
    "for",
    "from",
    "give",
    "had",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "might",
    "more",
    "most",
    "not",
    "of",
    "on",
    "or",
    "our",
    "page",
    "pages",
    "pdf",
    "please",
    "show",
    "such",
    "tell",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "topics",
    "use",
    "using",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
    "your",
}


class PDFProcessingError(ValueError):
    """Raised when a PDF cannot be validated, read, or rendered."""


def _check_pdf_bytes(pdf_bytes: bytes) -> bytes:
    """Check basic byte rules before handing the data to PyMuPDF."""
    if not isinstance(pdf_bytes, (bytes, bytearray)):
        raise PDFProcessingError("The uploaded file does not contain PDF data.")

    if len(pdf_bytes) == 0:
        raise PDFProcessingError("The uploaded PDF is empty.")

    if len(pdf_bytes) > MAX_PDF_BYTES:
        raise PDFProcessingError(
            f"The PDF is too large. Maximum size is "
            f"{MAX_PDF_BYTES // (1024 * 1024)} MB."
        )

    return bytes(pdf_bytes)


def _open_pdf(pdf_bytes: bytes) -> pymupdf.Document:
    """Open and validate a PDF. The caller must close the document."""
    checked_bytes = _check_pdf_bytes(pdf_bytes)

    try:
        # Opening from a stream keeps the PDF in memory instead of saving a
        # private document to disk.
        document = pymupdf.open(stream=checked_bytes, filetype="pdf")
    except Exception as error:
        # PyMuPDF uses several exception types for damaged files.
        raise PDFProcessingError(
            "The file is not a valid PDF or it is corrupted."
        ) from error

    try:
        if document.needs_pass or document.is_encrypted:
            raise PDFProcessingError(
                "This PDF is encrypted or password-protected. "
                "Please upload a PDF that opens without a password."
            )

        if document.page_count == 0:
            raise PDFProcessingError("The PDF does not contain any pages.")

        if document.page_count > MAX_PDF_PAGES:
            raise PDFProcessingError(
                f"The PDF has {document.page_count} pages. "
                f"Maximum supported length is {MAX_PDF_PAGES} pages."
            )

        return document
    except Exception:
        document.close()
        raise


def validate_pdf(pdf_bytes: bytes) -> None:
    """Validate a PDF without keeping the document open."""
    document = _open_pdf(pdf_bytes)
    document.close()


def get_pdf_info(pdf_bytes: bytes) -> dict[str, object]:
    """Return basic information about a valid PDF."""
    document = _open_pdf(pdf_bytes)
    try:
        metadata = document.metadata or {}
        return {
            "page_count": document.page_count,
            "size_bytes": len(pdf_bytes),
            "title": metadata.get("title", "") or "",
            "mime_type": "application/pdf",
        }
    finally:
        document.close()


def _normalize_text(raw_text: str) -> str:
    """Remove blank lines and unnecessary spaces while keeping line order."""
    cleaned_lines = []
    for line in raw_text.splitlines():
        cleaned_line = " ".join(line.split())
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    return "\n".join(cleaned_lines)


def extract_pdf_text(pdf_bytes: bytes) -> list[dict[str, object]]:
    """Extract selectable text page by page.

    Returns one dictionary per page with a 1-based page number. The has_text
    flag is False for scanned or image-only pages.
    """
    document = _open_pdf(pdf_bytes)

    try:
        pages = []
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            text = _normalize_text(page.get_text("text"))
            pages.append(
                {
                    "page_number": page_index + 1,
                    "text": text,
                    "has_text": bool(text),
                    "char_count": len(text),
                }
            )
        return pages
    except PDFProcessingError:
        raise
    except Exception as error:
        raise PDFProcessingError(
            "The PDF text could not be extracted. The document may be corrupted."
        ) from error
    finally:
        document.close()


def render_pdf_page(pdf_bytes: bytes, page_number: int) -> bytes:
    """Render one 1-based PDF page as PNG bytes at 144 DPI."""
    if isinstance(page_number, bool) or not isinstance(page_number, int):
        raise PDFProcessingError("Page number must be a whole number.")

    document = _open_pdf(pdf_bytes)

    try:
        if page_number < 1 or page_number > document.page_count:
            raise PDFProcessingError(
                f"Page {page_number} is outside this PDF "
                f"(valid range: 1 to {document.page_count})."
            )

        page = document.load_page(page_number - 1)
        width = page.rect.width * RENDER_DPI / 72
        height = page.rect.height * RENDER_DPI / 72

        if width <= 0 or height <= 0:
            raise PDFProcessingError("The selected page has invalid dimensions.")

        if width * height > MAX_RENDER_PIXELS:
            raise PDFProcessingError(
                "The selected page is too large to render safely at this resolution."
            )

        # alpha=False gives an opaque page image, which is better for screenshots.
        pixmap = page.get_pixmap(dpi=RENDER_DPI, alpha=False)
        return pixmap.tobytes("png")
    except PDFProcessingError:
        raise
    except Exception as error:
        raise PDFProcessingError(
            "The selected page could not be rendered. The PDF may be corrupted."
        ) from error
    finally:
        document.close()


def _question_keywords(question: str) -> set[str]:
    """Return useful lowercase words for simple keyword matching."""
    words = re.findall(r"[a-zA-Z0-9]+", question.lower())
    return {word for word in words if len(word) >= 3 and word not in STOPWORDS}


def _question_page_references(question: str, page_count: int) -> list[int]:
    """Read explicit references such as 'page 4' from the question."""
    references = []
    for match in re.findall(r"\bpage\s+(\d+)\b", question.lower()):
        page_number = int(match)
        if page_number < 1 or page_number > page_count:
            raise PDFProcessingError(
                f"Page {page_number} does not exist. The PDF has "
                f"{page_count} pages."
            )
        if page_number not in references:
            references.append(page_number)
    return references


def select_relevant_pages(
    pages: list[dict[str, object]],
    question: str,
    max_pages: int = MAX_RELEVANT_PAGES,
) -> list[dict[str, object]]:
    """Choose a few pages using page references and simple keyword scoring.

    This is a lightweight lexical heuristic, not semantic retrieval. If the
    question has no usable keywords and the document is too large, the caller
    should ask the user to choose a page manually.
    """
    if not pages:
        raise PDFProcessingError("The PDF does not contain any pages to search.")

    if not isinstance(question, str) or not question.strip():
        raise PDFProcessingError("Please enter a question about the PDF.")

    max_pages = max(1, min(max_pages, MAX_RELEVANT_PAGES))
    page_count = len(pages)

    # Explicit "page N" requests always win over keyword scores.
    selected_numbers = _question_page_references(question, page_count)
    if len(selected_numbers) > max_pages:
        raise PDFProcessingError(
            f"A whole-document question can use at most {max_pages} pages. "
            "Please select one page or ask about fewer pages."
        )

    keywords = _question_keywords(question)
    raw_words = set(re.findall(r"[a-zA-Z0-9]+", question.lower()))

    scored_pages = []
    for page in pages:
        page_number = int(page["page_number"])
        if page_number in selected_numbers:
            continue

        text = str(page["text"]).lower()
        score = sum(text.count(keyword) for keyword in keywords)
        if score > 0:
            scored_pages.append((score, page_number))

    # Highest score first; ties use the earlier page for a stable result.
    scored_pages.sort(key=lambda item: (-item[0], item[1]))

    for _score, page_number in scored_pages:
        if len(selected_numbers) >= max_pages:
            break
        if page_number not in selected_numbers:
            selected_numbers.append(page_number)

    # A general summary has no specific matching words. Use the first few pages
    # and clearly tell the model that only those pages were supplied.
    if not selected_numbers and raw_words & SUMMARY_WORDS:
        selected_numbers = list(range(1, min(page_count, max_pages) + 1))

    if not selected_numbers:
        raise PDFProcessingError(
            "Could not confidently match the question to specific pages. "
            "Use Selected page mode and choose a page manually."
        )

    selected_numbers.sort()
    pages_by_number = {int(page["page_number"]): page for page in pages}
    return [pages_by_number[number] for number in selected_numbers]
