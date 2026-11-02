"""Image validation and preprocessing for image question answering.

The uploaded bytes are never modified. Each function opens its own copy from
memory and closes it when finished.
"""

from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

# The upload limit applies to the original file, before resizing.
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 40_000_000
MAX_DIMENSION = 1600

# Pillow reports the detected format, not the file extension.
SUPPORTED_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class ImageValidationError(ValueError):
    """Raised when an uploaded image cannot be accepted or processed."""


def _check_image_bytes(image_bytes: bytes) -> bytes:
    """Check the basic byte-size rules before asking Pillow to decode."""
    if not isinstance(image_bytes, (bytes, bytearray)):
        raise ImageValidationError("The uploaded file does not contain image data.")

    if len(image_bytes) == 0:
        raise ImageValidationError("The uploaded image is empty.")

    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ImageValidationError(
            f"The image is too large. Maximum size is "
            f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB."
        )

    return bytes(image_bytes)


def _open_image(image_bytes: bytes) -> Image.Image:
    """Open, decode, and check an image. The caller must close the image."""
    checked_bytes = _check_image_bytes(image_bytes)

    try:
        image = Image.open(BytesIO(checked_bytes))
    except Image.DecompressionBombError as error:
        raise ImageValidationError("The image dimensions are too large.") from error
    except UnidentifiedImageError as error:
        raise ImageValidationError(
            "The file is not a valid image or its format is not supported."
        ) from error
    except (OSError, ValueError, SyntaxError) as error:
        raise ImageValidationError("The image could not be opened.") from error

    try:
        image_format = image.format
        if image_format not in SUPPORTED_FORMATS:
            raise ImageValidationError(
                "Unsupported image format. Please upload a JPEG, PNG, or WEBP image."
            )

        width, height = image.size
        if width <= 0 or height <= 0:
            raise ImageValidationError("The image has invalid dimensions.")

        if width * height > MAX_PIXELS:
            raise ImageValidationError(
                "The image resolution is too large. Please upload a smaller image."
            )

        # load() forces Pillow to decode the pixels and catches truncated files.
        image.load()
        return image
    except ImageValidationError:
        image.close()
        raise
    except (OSError, ValueError, SyntaxError) as error:
        image.close()
        raise ImageValidationError(
            "The image is corrupted or could not be decoded."
        ) from error


def validate_image(image_bytes: bytes) -> None:
    """Validate an uploaded image without keeping it in memory.

    Raises:
        ImageValidationError: If the file is empty, too large, corrupted, or
            not a supported JPEG, PNG, or WEBP image.
    """
    image = _open_image(image_bytes)
    image.close()


def get_image_info(image_bytes: bytes) -> dict[str, object]:
    """Return basic information about a valid image."""
    image = _open_image(image_bytes)
    try:
        info: dict[str, object] = {
            "format": image.format,
            "mime_type": SUPPORTED_FORMATS[image.format],
            "width": image.width,
            "height": image.height,
            "size_bytes": len(image_bytes),
        }
        return info
    finally:
        image.close()


def _convert_to_rgb(image: Image.Image) -> Image.Image:
    """Return a new RGB image, putting transparent areas on a white background."""
    if image.mode == "RGB":
        return image.copy()

    has_transparency = image.mode in ("RGBA", "LA") or "transparency" in image.info
    if has_transparency:
        rgba_image = image.convert("RGBA")
        try:
            # A white background keeps text readable when alpha is removed.
            background = Image.new("RGB", rgba_image.size, (255, 255, 255))
            background.paste(rgba_image, mask=rgba_image.getchannel("A"))
            return background
        finally:
            rgba_image.close()

    return image.convert("RGB")


def prepare_image(image_bytes: bytes) -> tuple[bytes, str]:
    """Validate, orient, resize, and convert an image for the vision model.

    Returns:
        A tuple of (processed image bytes, MIME type).

    Raises:
        ImageValidationError: If the image cannot be validated or processed.
    """
    image = _open_image(image_bytes)

    try:
        source_format = image.format

        # EXIF transpose fixes photos that are rotated by the camera.
        oriented = ImageOps.exif_transpose(image)
        try:
            rgb_image = _convert_to_rgb(oriented)
        finally:
            # exif_transpose can return a new image, so close it separately.
            if oriented is not image:
                oriented.close()

        try:
            # thumbnail() resizes in place and keeps the aspect ratio.
            rgb_image.thumbnail(
                (MAX_DIMENSION, MAX_DIMENSION),
                Image.Resampling.LANCZOS,
            )

            output = BytesIO()
            if source_format == "JPEG":
                rgb_image.save(output, format="JPEG", quality=90, optimize=True)
            elif source_format == "PNG":
                rgb_image.save(output, format="PNG", optimize=True)
            else:
                rgb_image.save(output, format="WEBP", quality=90)

            return output.getvalue(), SUPPORTED_FORMATS[source_format]
        finally:
            rgb_image.close()
    except ImageValidationError:
        raise
    except (OSError, ValueError) as error:
        raise ImageValidationError("The image could not be prepared.") from error
    finally:
        image.close()
