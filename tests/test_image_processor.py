"""Tests for image validation and preprocessing."""

from io import BytesIO

import pytest
from PIL import Image

from modules.image_processor import (
    MAX_IMAGE_BYTES,
    ImageValidationError,
    get_image_info,
    prepare_image,
    validate_image,
)


def make_image_bytes(
    image_format: str = "PNG",
    size: tuple[int, int] = (120, 80),
    mode: str = "RGB",
    color: object = "red",
) -> bytes:
    """Create a small image in memory for a test."""
    buffer = BytesIO()
    Image.new(mode, size, color).save(buffer, format=image_format)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("image_format", "mime_type"),
    [
        ("PNG", "image/png"),
        ("JPEG", "image/jpeg"),
        ("WEBP", "image/webp"),
    ],
)
def test_valid_images_have_correct_mime_type(image_format, mime_type):
    image_bytes = make_image_bytes(image_format)

    prepared_bytes, prepared_mime = prepare_image(image_bytes)

    assert prepared_mime == mime_type
    assert len(prepared_bytes) > 0
    with Image.open(BytesIO(prepared_bytes)) as prepared_image:
        assert prepared_image.mode == "RGB"


def test_get_image_info_returns_dimensions_and_format():
    image_bytes = make_image_bytes("PNG", size=(320, 240))

    info = get_image_info(image_bytes)

    assert info["format"] == "PNG"
    assert info["mime_type"] == "image/png"
    assert info["width"] == 320
    assert info["height"] == 240
    assert info["size_bytes"] == len(image_bytes)


def test_invalid_bytes_are_rejected():
    with pytest.raises(ImageValidationError, match="not a valid image"):
        validate_image(b"this is not an image")


def test_empty_image_is_rejected():
    with pytest.raises(ImageValidationError, match="empty"):
        validate_image(b"")


def test_oversized_file_is_rejected():
    oversized_bytes = b"\x00" * (MAX_IMAGE_BYTES + 1)

    with pytest.raises(ImageValidationError, match="too large"):
        validate_image(oversized_bytes)


def test_unsupported_format_is_rejected():
    gif_bytes = make_image_bytes("GIF")

    with pytest.raises(ImageValidationError, match="Unsupported image format"):
        validate_image(gif_bytes)


def test_large_image_is_resized_and_keeps_aspect_ratio():
    image_bytes = make_image_bytes("PNG", size=(3000, 1000))

    prepared_bytes, _ = prepare_image(image_bytes)

    with Image.open(BytesIO(prepared_bytes)) as prepared_image:
        assert prepared_image.size == (1600, 533)


def test_exif_orientation_is_applied():
    image = Image.new("RGB", (300, 200), "blue")
    exif = Image.Exif()
    exif[274] = 6  # EXIF orientation 6 rotates the image 90 degrees.
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)

    prepared_bytes, _ = prepare_image(buffer.getvalue())

    with Image.open(BytesIO(prepared_bytes)) as prepared_image:
        assert prepared_image.size == (200, 300)


def test_transparent_png_is_flattened_to_rgb():
    image_bytes = make_image_bytes(
        "PNG",
        size=(50, 50),
        mode="RGBA",
        color=(255, 0, 0, 0),
    )

    prepared_bytes, _ = prepare_image(image_bytes)

    with Image.open(BytesIO(prepared_bytes)) as prepared_image:
        assert prepared_image.mode == "RGB"
        assert prepared_image.getpixel((0, 0)) == (255, 255, 255)
